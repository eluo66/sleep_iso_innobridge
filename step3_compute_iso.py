import os, pickle, subprocess, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne
import pyedflib
from compute_iso import get_iso
    


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

    output_dir = 'result'
    os.makedirs(output_dir, exist_ok=True)
    epoch_time = 30
    # we need flexible channel names
    # we can implement this using regular expression (re)
    target_ch_names = ['F3-[AM]2', 'F4-[AM]1', 'C3-[AM]2', 'C4-[AM]1', 'O1-[AM]2', 'O2-[AM]1']
    standard_ch_names = ['F3M2', 'F4M1', 'C3M2', 'C4M1', 'O1M2', 'O2M1']#, 'ECG']
    ch_groups = [['F3M2', 'F4M1'], ['C3M2', 'C4M1'], ['O1M2', 'O2M1']]
    ch_group_names =['Frontal', 'Central', 'Occipital'] #EEG channels are grouped into frontal, central, and occipital regions for regional analysis

    all_results = []
    for i in tqdm(range(len(df))): #loops through each person (row in the mastersheet)
        try:
            sid = f'{df.SiteID.iloc[i]}-{df.BDSPPatientID.iloc[i]}-{df.SessionID.iloc[i]}'

            edf_path, annot_path = get_path(df.BidsFolder.iloc[i], df.SessionID.iloc[i])

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
            sleep_stages = np.array([mapping.get(x, np.nan) for x in sleep_stages])

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

            # align sleep stages and EEG
            T_eeg = eeg.shape[1]/sfreq
            T_sleep_stages = len(sleep_stages)*30
            Tmin = min(T_eeg, T_sleep_stages)
            eeg = eeg[:,:int(Tmin*sfreq)]
            sleep_stages = sleep_stages[:Tmin//30]

            # filter to remove the unwanted signals, removes 60 Hz powerline noise, only keeps 0.3-35 frequencies
            notch_freq = 60  # Hz
            eeg = eeg-np.nanmean(eeg,axis=-1,keepdims=True)
            eeg = mne.filter.notch_filter(eeg, sfreq, notch_freq, verbose=False)
            eeg = mne.filter.filter_data(eeg, sfreq, 0.3, 35, verbose=False)

            # run ISO detection
            res, spec_iso, freq_iso, bp_signals, sfreq_bp_signal = get_iso(eeg, sfreq, standard_ch_names,
                                                                    sleep_stages, ch_groups, ch_group_names)
            # res: summary metrics per channel or group
            # spec_iso: spectrograms or power spectra for ISOs
            # freq_iso: frequencies analyzed
            # bp_signals: filtered bandpassed signals
            # sfreq_bp_signal: sampling frequency of bandpassed signal
            
            # save results
            with open(os.path.join(output_dir, f'ISO_result_{sid}.pickle'), 'wb') as f:
                pickle.dump((spec_iso, freq_iso, bp_signals, sfreq_bp_signal), f)

            res['SID'] = sid
            all_results.append(res)
        
            all_results_now = pd.DataFrame(data=all_results)
            print(all_results_now)
            all_results_now.to_csv(os.path.join(output_dir, 'ISO_features.csv'), index=False)

            # to save disk space
            if os.path.exists(edf_path):
                os.remove(edf_path)
            if os.path.exists(annot_path):
                os.remove(annot_path)
        except Exception as e:
            print(f'{sid}: ERROR {e}')




if __name__=='__main__':
    main()
