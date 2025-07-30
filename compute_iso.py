from collections import defaultdict
from itertools import groupby, product
import os, subprocess, pickle
#import emd
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, mode
from scipy.interpolate import interp1d
from scipy.signal import detrend, resample, find_peaks_cwt, savgol_filter, find_peaks
from tqdm import tqdm
import pyedflib
import mne
from mne.time_frequency import tfr_array_morlet, psd_array_multitaper
from specparam import SpectralModel


def get_iso(eeg, sfreq, ch_names, sleep_stages, ch_groups, ch_group_names):#, artifact_indicator_
    """
    eeg.shape = (#channel, #points)
    sfreq: sampling frequency of EEG in Hz
    ch_names: a list of channel names
    sleep_stages.shape = (#30s-epochs,)
    # not used: artifact_indicator
    ch_groups: [['C3-M2', 'C4-M1'], ....]
    ch_group_names: ['Central', ...]
    """
    # Lázár, Z.I., Dijk, D.J. and Lázár, A.S., 2019.
    # Infraslow oscillations in human sleep spindle activity. Journal of neuroscience methods, 316, pp.22-34.
    # resample to 128Hz
    eeg = resample(eeg, int(round(eeg.shape[1]/sfreq*128)), axis=-1)
    sfreq = 128

    """
    # remove transient artifacts that lasts shortly to maximize continuous N2+N3
    artifact_indicator = np.array(artifact_indicator_)
    for i in range(artifact_indicator_.shape[1]):
        cc = 0
        for k,l in groupby(artifact_indicator_[:,i]):
            ll = len(list(l))
            if k and ll<=2:
                artifact_indicator[cc,i] = False
            cc += ll
    # remove transient N1 that lasts shortly to maximize continuous N2+N3
    sleep_stages = np.array(sleep_stages_)
    n2n3 = np.in1d(sleep_stages_,[1,2])
    cc = 0
    for k,l in groupby(n2n3):
        ll = len(list(l))
        if not k and ll<=2 and (sleep_stages_[cc:cc+ll]==3).all():
            sleep_stages[cc:cc+ll] = 2
        cc += ll
        
    artifact_indicator = np.array([artifact_indicator[:,[ch_names.index(x) for x in xx]].any(axis=1) for xx in ch_groups])
    """

    window_size = int(round(8*sfreq))
    step_size = int(round(2*sfreq))
    epoch_time = 2
    sfreq_bp_signal = 1/epoch_time
    start_ids = np.arange(0, eeg.shape[1]-window_size+1, step_size)
    epochs = np.array([eeg[:,x:x+window_size] for x in start_ids])
    #epochs = detrend(epochs, axis=-1)
    epochs = epochs - epochs.mean(axis=-1,keepdims=True)

    # make sleep_stages, artifact_indicator the same length as signal
    sleep_stages = np.repeat(sleep_stages, int(round(30*sfreq)))
    assert len(sleep_stages)==eeg.shape[1]
    sleep_stages = np.array([mode(sleep_stages[x:x+window_size], keepdims=False).mode for x in start_ids])
    """
    artifact_indicator = np.repeat(artifact_indicator, int(round(30*sfreq)), axis=1)
    assert artifact_indicator.shape[1]==eeg.shape[1]
    artifact_indicator = np.array([artifact_indicator[:,x:x+window_size].any(axis=1) for x in start_ids])
    """

    spec, freq = psd_array_multitaper(epochs, sfreq=sfreq, fmin=1, fmax=30, bandwidth=2, normalization='full', verbose=False)

    dfreq = freq[1]-freq[0]
    spec[np.isinf(spec)] = np.nan
    
    window_size = int(round(512*sfreq_bp_signal))
    step_size = int(round(64*sfreq_bp_signal))
    freq_iso = np.linspace(0,0.1,101)[1:]
    spec_iso_all_ch = {}
    bp_signals = {}
    res = {}
    band_names = ['alpha', 'delta', 'sigma_oof', 'sigma', 'slow sigma', 'fast sigma', 'oof']
    freq_ranges = [[8,12],[1,4],[11,15], [11,15], [11,13], [13,15], None]
    for bn, freq_range in zip(band_names, freq_ranges):
        if bn=='oof':
            bp_signal = np.polyfit(np.log(freq), np.log(spec.reshape(-1, spec.shape[-1])).T, deg=1)[0]
            bp_signal = bp_signal.reshape(-1,len(ch_names)).T
        elif bn=='sigma_oof':
            #TODO ff = SpectralGroupModel()
            #ff.fit(freq, spec.reshape(-1,spec.shape[-1]))
            spec2 = spec.reshape(-1, spec.shape[-1])
            coef, b0 = np.polyfit(np.log(freq), np.log(spec2).T, deg=1)
            freq_ids = (freq>=freq_range[0])&(freq<=freq_range[1])
            bp_signal = np.sum(spec2[:,freq_ids]-np.exp(np.log(freq[freq_ids].reshape(-1,1))*coef+b0).T, axis=1)
            bp_signal = bp_signal.reshape(-1,len(ch_names)).T
        else:
            freq_ids = (freq>=freq_range[0])&(freq<=freq_range[1])
            bp_signal = 10*np.log10(np.sum(spec[...,freq_ids], axis=-1)*dfreq).T  # (#ch, #epoch)

        bp_signal = np.array([np.nanmean(bp_signal[[ch_names.index(x) for x in xx]], axis=0) for xx in ch_groups])
        bp_signals[bn] = bp_signal
        
        spec_iso_all_ch[bn] = []
        for chi,ch in enumerate(ch_group_names):
            good_ids = np.in1d(sleep_stages,[1,2])#&(~artifact_indicator[:,chi])
            spec_isos = []
            cc = 0
            for k,l in groupby(good_ids):
                ll = len(list(l))
                if not k:
                    cc += ll
                    continue
                for start in np.arange(cc, cc+ll-window_size+1, step_size):
                    xx = bp_signal[chi,start:start+window_size]
                    #xx = detrend(xx, axis=-1)
                    xx = xx-xx.mean(axis=-1,keepdims=True)
                    
                    # use EMD to denoise
                    #imf = emd.sift.sift(xx)
                    #xx = imf[:,1:-1].sum(axis=-1)
                        
                    spec_iso, _ = psd_array_multitaper(xx, sfreq_bp_signal, fmin=0, fmax=0.2,
                            bandwidth=0.01, normalization='full', verbose=False)
                    # interpolate to make same length
                    ff = interp1d(_, spec_iso, axis=-1)
                    spec_iso = ff(freq_iso)
                    # make relative
                    spec_iso = spec_iso/spec_iso.sum(axis=-1,keepdims=True)

                    spec_isos.append(spec_iso)
                cc += ll
            if len(spec_isos)==0:
                res[f'ISO_peak_freq_{bn}_{ch}'] = np.nan
                res[f'ISO_peak_relpower_{bn}_{ch}'] = np.nan
                res[f'ISO_bandpower_{bn}_{ch}'] = np.nan
            else:
                spec_iso = np.nanmean(np.array(spec_isos), axis=0)
                x = savgol_filter(spec_iso, 10, 2)
                peak_idx, _ = find_peaks(x)
                if len(peak_idx)>0:
                    res[f'ISO_peak_freq_{bn}_{ch}'] = freq_iso[peak_idx[0]]
                    res[f'ISO_peak_relpower_{bn}_{ch}'] = x[peak_idx[0]]
                    res[f'ISO_bandpower_{bn}_{ch}'] = spec_iso[(freq_iso>=0.005)&(freq_iso<=0.03)].sum() # not *dfreq since relative power
                else:
                    res[f'ISO_peak_freq_{bn}_{ch}'] = np.nan
                    res[f'ISO_peak_relpower_{bn}_{ch}'] = np.nan
                    res[f'ISO_bandpower_{bn}_{ch}'] = np.nan
                spec_iso_all_ch[bn].append(spec_iso)

    spec_iso = {k:np.array(v) for k,v in spec_iso_all_ch.items()}
    return res, spec_iso, freq_iso, bp_signals, sfreq_bp_signal


