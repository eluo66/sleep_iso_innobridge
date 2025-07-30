import numpy as np
import pandas as pd
#from pyedflib.highlevel import read_edf
import mne
import xmltodict


def load_dataset(edf_path, annot_path):
    """
    signals
    sleep_stages: array of sleep stage every 30 seconds, aligned with signals, W 5, R 4, N1 3, N2 2, N1 1, otherwise -1
    params
    """
    #signals, sig_hdrs, hdr = read_edf(edf_path)
    edf = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    ch_names = edf.ch_names
    params = dict(edf.info)

    # load annotation
    with open(annot_path,'r') as f:
        annot = xmltodict.parse(f.read())
    assert int(annot['CMPStudyConfig']['EpochLength'])==30##
    ss_mapping = {0:5,1:3,2:2,3:1,4:1,5:4}
    sleep_stages = np.array([ss_mapping.get(int(x),np.nan) for x in annot['CMPStudyConfig']['SleepStages']['SleepStage']])
    annot = pd.DataFrame(data=annot['CMPStudyConfig']['ScoredEvents']['ScoredEvent'])
    params['annot'] = annot

    # load EEG
    eeg_ch_names1 = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'A1', 'A2']
    eeg_ch_names2 = ['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1']
    if np.in1d(eeg_ch_names1, ch_names).all():
        edf_ = mne.io.read_raw_edf(edf_path, preload=True, exclude=[x for x in ch_names if x not in eeg_ch_names1], verbose=False)
        eeg = edf_.get_data(picks=eeg_ch_names1)
        eeg = np.array([
            eeg[eeg_ch_names1.index('F3')] - eeg[eeg_ch_names1.index('A2')],
            eeg[eeg_ch_names1.index('F4')] - eeg[eeg_ch_names1.index('A1')],
            eeg[eeg_ch_names1.index('C3')] - eeg[eeg_ch_names1.index('A2')],
            eeg[eeg_ch_names1.index('C4')] - eeg[eeg_ch_names1.index('A1')],
            eeg[eeg_ch_names1.index('O1')] - eeg[eeg_ch_names1.index('A2')],
            eeg[eeg_ch_names1.index('O2')] - eeg[eeg_ch_names1.index('A1')],
            ])
    elif np.in1d(eeg_ch_names2, ch_names).all():
        edf_ = mne.io.read_raw_edf(edf_path, preload=True, exclude=[x for x in ch_names if x not in eeg_ch_names2], verbose=False)
        eeg = edf_.get_data(picks=eeg_ch_names2)
    else:
        raise ValueError(f'EEG not found, channels = {ch_names} in {edf_path}')
    sfreq_eeg = edf_.info['sfreq']
    eeg *= 1e6
    eeg_ch_names2 = ['F3M2', 'F4M1', 'C3M2', 'C4M1', 'O1M2', 'O2M1']

    # load ECG
    ecg_ch_names1 = ['ECG1', 'ECG2']
    ecg_ch_names2 = ['ECG1-ECG2']
    if np.in1d(ecg_ch_names1, ch_names).all():
        edf_ = mne.io.read_raw_edf(edf_path, preload=True, exclude=[x for x in ch_names if x not in ecg_ch_names1], verbose=False)
        ecg = edf_.get_data(picks=ecg_ch_names1)
        ecg = ecg[ecg_ch_names1.index('ECG1')] - ecg[ecg_ch_names1.index('ECG2')]
    elif np.in1d(ecg_ch_names2, ch_names).all():
        edf_ = mne.io.read_raw_edf(edf_path, preload=True, exclude=[x for x in ch_names if x not in ecg_ch_names2], verbose=False)
        ecg = edf_.get_data(picks=ecg_ch_names2)
        ecg = ecg[0]
    else:
        raise ValueError(f'ECG not found, channels = {ch_names} in {edf_path}')
    sfreq_ecg = edf_.info['sfreq']
    ecg *= 1e6
    ecg_ch_names2 = ['ECG']

    #TODO load EOG, EMG, ...

    # align with sleep stages, by making it multiplicate of 30 seconds
    Teeg = int((eeg.shape[-1]/sfreq_eeg)//30)*30
    Tecg = int((ecg.shape[-1]/sfreq_ecg)//30)*30
    T = len(sleep_stages)*30
    L = min(Teeg,Tecg,T)
    sleep_stages = sleep_stages[:L//30]
    eeg = eeg[...,:int(round(L*sfreq_eeg))]
    ecg = ecg[...,:int(round(L*sfreq_ecg))]

    params['ch_names'] = eeg_ch_names2+ecg_ch_names2
    params['meas_date'] = params['meas_date'].replace(tzinfo=None)
    params['sfreq_eeg'] = sfreq_eeg
    params['sfreq_ecg'] = sfreq_ecg
    return eeg, ecg, sleep_stages, params

