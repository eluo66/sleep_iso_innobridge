import pandas as pd
import numpy as np

"""
aws s3 ls s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-psg-access-point/PSG/bids/S0001/sub-S0001111189075/ses-1/eeg/
2025-02-27 02:24:43      41280 sub-S0001111189075_ses-1_task-psg_annotations.csv
2025-02-27 02:24:43       6239 sub-S0001111189075_ses-1_task-psg_channels.tsv
2025-02-27 02:24:43 2998141350 sub-S0001111189075_ses-1_task-psg_eeg.edf
2025-02-27 02:24:43        472 sub-S0001111189075_ses-1_task-psg_eeg.json
"""


# get metadata, that include age and sex
df = pd.read_csv('bdsp_psg_master_20231101.csv')
print(f'Original shape is {len(df)}')

# apply inclusion criteria
# >=18 years
df = df[df.AgeAtVisit>=18].reset_index(drop=True)
print(f'After >=18y, shape is {len(df)}')


# diagnostic
df = df[df.StudyType.astype(str).str.contains('dia', case=False)].reset_index(drop=True)
print(f'After diagnostic only, shape is {len(df)}')

np.random.seed(16)

# drop duplicates
df = df.iloc[np.random.choice(len(df), len(df), replace=False)].reset_index(drop=True)
df = df.drop_duplicates('BDSPPatientID', ignore_index=True)

# to reduce sample size, subsample 1000 patients
# make sure we have even distribution of age and sex among the 1000 patients
# age bins: 18-30, 30-50, 50-70, 70-95 (4)
# sex bins: female and male (2)
# therefore 125 patients / bin


Ntarget = 1000
age_bins = [[18,30], [30,50], [50,70], [70,96]]
sex_bins = ['Female', 'Male']
Ntarget_bin = Ntarget // len(age_bins) // len(sex_bins)
df_res = []
for age in age_bins:
    for sex in sex_bins:
        df_bin = df[(df.AgeAtVisit>=age[0]) & (df.AgeAtVisit<age[1]) & (df.SexDSC==sex)]
        # randomly sample within each bin
        random_ids = np.random.choice(len(df_bin), Ntarget_bin, replace=False)
        df_bin = df_bin.iloc[random_ids]
        df_res.append(df_bin)

df_res = pd.concat(df_res, axis=0, ignore_index=True)
print(df_res)
assert len(df_res)==Ntarget
assert df_res.BDSPPatientID.nunique()==Ntarget

# save the mastersheet: SiteID, SID, SessionID, age, sex, StudyType, AHI, PLMI, EDFPath, AnnotPath
df_res.to_excel('mastersheet-full.xlsx', index=False)
