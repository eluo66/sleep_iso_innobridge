from itertools import groupby
import datetime, os, subprocess, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import peak_prominences
import pyedflib
import neurokit2 as nk
import mne
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import seaborn as sns
sns.set_style('ticks')

#Draws the spectrogram and sleep stage timeline
# sleep_stages = sleep label for each time chunk (like N3, N2, etc.).
# spec_db = brainwave power across different frequencies and channels.
#freq = frequency values (like 1 Hz to 20 Hz).
#tt = time points (in hours).
#start_time = when the recording started.
#ch_names = EEG channel labels (like "Frontal", "Central", etc.).
def plot(sleep_stages, spec_db, freq, tt, start_time, ch_names, artifact_indicator=None, fig_path=None):
    vmin, vmax = -5, 25
    xlim = [tt.min(), tt.max()]
    xticks = np.arange(0, int(tt.max())+1)
    xticklabels = [(start_time+datetime.timedelta(hours=int(x))).strftime('%H:%M') for x in xticks]
    Nch = spec_db.shape[1]

    plt.close()
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1+Nch, height_ratios=[1]+[2]*Nch, hspace=0.1)

    ax = fig.add_subplot(gs[0]); ax0 = ax
    ax.step(tt, sleep_stages, where='post', c='k')
    ax.set_yticks([1,2,3,4,5], labels=['N3','N2','N1','R','W'])
    ax.set_ylim(0.9,5.1)
    ax.yaxis.grid(True)
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_xlim(xlim)
    plt.setp(ax.get_xticklabels(), visible=False)

    for chi in range(Nch):
        ax = fig.add_subplot(gs[1+chi], sharex=ax0)
        ax.imshow(spec_db[:,chi].T, aspect='auto', origin='lower', #uses color to show power
                  cmap='turbo', vmin=vmin, vmax=vmax,
                  extent=(tt.min(), tt.max(), freq.min(), freq.max()))
        if artifact_indicator is not None:
            es = artifact_indicator[:,chi].astype(float)
            es[es==0] = np.nan
            ax.step(tt, es*freq.max()+0.5, where='post', color='k', lw=3)
        ax.set_ylim(freq.min(), freq.max()+1)
        ax.set_ylabel(ch_names[chi])
        ax.set_xticks(xticks, labels=xticklabels)
        ax.set_xlim(xlim)
        if chi<Nch-1:
            plt.setp(ax.get_xticklabels(), visible=False)

    plt.tight_layout()
    if fig_path is None:
        plt.show()
    else:
        plt.savefig(fig_path, bbox_inches='tight')


def get_path(bids_folder, session):
    tmp_folder = 'local_tmp'
    os.makedirs(tmp_folder, exist_ok=True)

    # download EDF file
    edf_path = os.path.join(tmp_folder, f'{bids_folder}_ses-{session}_task-psg_eeg.edf')
    if not os.path.exists(edf_path):
        cmd = ['aws', 's3', 'cp',
        f's3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-psg-access-point/PSG/bids/S0001/{bids_folder}/ses-{session}/eeg/{bids_folder}_ses-{session}_task-psg_eeg.edf',
        edf_path]
        subprocess.run(cmd)

    # download annotation file
    annot_path = os.path.join(tmp_folder, f'{bids_folder}_ses-{session}_task-psg_annotations.csv')
    if not os.path.exists(annot_path):
        cmd = ['aws', 's3', 'cp',
        f's3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-psg-access-point/PSG/bids/S0001/{bids_folder}/ses-{session}/eeg/{bids_folder}_ses-{session}_task-psg_annotations.csv',
        annot_path]
        subprocess.run(cmd)

    return edf_path, annot_path



def main():
    df = pd.read_excel('mastersheet-full.xlsx')
    base_dir = '.'

    fig_dir = 'figure_spectrogram'
    os.makedirs(fig_dir, exist_ok=True)
    output_dir = 'clean_edf'
    os.makedirs(output_dir, exist_ok=True)
    spec_dir = 'spectrogram'
    os.makedirs(spec_dir, exist_ok=True) #creates folders to save final figures, cleaned EEG files, spectrogram data
    epoch_time = 30

    # we need flexible channel names
    # we can implement this using regular expression (re)
    target_ch_names = ['F3-[AM]2', 'F4-[AM]1', 'C3-[AM]2', 'C4-[AM]1', 'O1-[AM]2', 'O2-[AM]1', 'E[CK]G']

    standard_ch_names = ['F3M2', 'F4M1', 'C3M2', 'C4M1', 'O1M2', 'O2M1', 'ECG']
    print(df)

    for i in tqdm(range(len(df))): #loops through each person (row in the mastersheet)
        sid = f'{df.SiteID.iloc[i]}-{df.BDSPPatientID.iloc[i]}-{df.SessionID.iloc[i]}'

        edf_path, annot_path = get_path(df.BidsFolder.iloc[i], df.SessionID.iloc[i])

        # load EDF file
        signals, signal_hdrs, hdr = pyedflib.highlevel.read_edf(edf_path)
        start_time = hdr['startdate']
        actual_ch_names = pd.DataFrame(signal_hdrs).label

        # to get EEG signals
        target_ch_ids = [[yid for yid, y in enumerate(actual_ch_names) if re.match(x,y)] for x in target_ch_names]
        target_ch_num = [len(x) for x in target_ch_ids]
        if min(target_ch_num)==0:
            continue
        if max(target_ch_num)>1:
            continue
        target_ch_ids = [x[0] for x in target_ch_ids]
        eeg = signals[target_ch_ids]

        # to get EEG signal sampling frequency (Hz)
        sfreq = signal_hdrs[0]['sample_frequency']  # 200Hz

        # filter to remove the unwanted signals, removes 60 Hz powerline noise, only keeps 0.3-35 frequencies
        notch_freq = 60  # Hz
        eeg = eeg-np.nanmean(eeg,axis=-1,keepdims=True)
        eeg = mne.filter.notch_filter(eeg, sfreq, notch_freq, verbose=False)
        eeg = mne.filter.filter_data(eeg, sfreq, 0.3, 35, verbose=False)

        """
        sleep_ids = np.where(np.in1d(sleep_stages,[1,2,3,4]))[0]
        start = int(round(sleep_ids[0]*epoch_time*sfreq_ecg))
        end   = int(round((sleep_ids[-1]+1)*epoch_time*sfreq_ecg))
        flip = determine_flip_ecg(ecg[start:end], sfreq_ecg)
        assert flip is not None
        if flip:
            ecg = -ecg
        """

        # get spectogram

        # spectrum: the power spectral density (PSD) at a certain frequency --> represent a signal in frequency domain
        # spectra: (plural) multiple spectrum
        # spectrogram: spectra across time --> represent the time varying frequency domain

        #splits EEG into 30 second pieces (epochs)

        window_size = int(round(epoch_time*sfreq))
        start_ids = np.arange(0, eeg.shape[1]-window_size+1, window_size)
        epochs = np.array([eeg[:,x:x+window_size] for x in start_ids])

        #Uses multitaper spectral estimation to get the power of each frequency in each time chunk
        #Then converts it to decibels (dB)
        spec, freq = mne.time_frequency.psd_array_multitaper(epochs, sfreq, fmin=0.5, fmax=20, bandwidth=0.5, normalization='full', verbose=False)
        spec_db  = 10*np.log10(spec)

        # add sleep stages
        # 5 W
        # 4 R
        # 3 N1
        # 2 N2
        # 1 N3

        sleep_stages = pd.read_csv(annot_path)
        # filter only events starting with Sleep_stage_
        sleep_stages = sleep_stages.event[sleep_stages.event.str.startswith('Sleep_stage_')]
        mapping = {'Sleep_stage_W': 5,
                   'Sleep_stage_R': 4,
                   'Sleep_stage_N1': 3,
                   'Sleep_stage_N2': 2,
                   'Sleep_stage_N3': 1,}  # anythingn else should be NaN (not a number)
        sleep_stages = [mapping.get(x, np.nan) for x in sleep_stages]

        # plot and save the spectrogram
        # tt: Time points (in hours)
        #spec_db_: Combines the 6 channels into 3 averaged regions
        #Then calls the plot() function to draw the final spectrogram with sleep stages
        fig_path = os.path.join(fig_dir, sid+'.png')
        tt = start_ids/sfreq/3600
        spec_db_ = np.nanmean(np.array([spec_db[:,::2],spec_db[:,1::2]]), axis=0)
        ch_names_ = ['Frontal', 'Central', 'Occipital']
        plot(sleep_stages, spec_db_, freq, tt, start_time, ch_names_)#, fig_path=fig_path)

        # to save disk space
        if os.path.exists(edf_path):
            os.remove(edf_path)
        if os.path.exists(annot_path):
            os.remove(annot_path)


if __name__=='__main__':
    main()

