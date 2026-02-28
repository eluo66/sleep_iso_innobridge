import numpy as np
import pandas as pd
from itertools import product


df=pd.read_csv('/data/haoqisun/AD_PD_prediction_from_sleep/MGH_dataset/MGH_covariates.csv')
print(f'After loading MGH_covariates: N={len(df)} rows, {df.BDSPPatientID.nunique()} unique patients')

df = df[
df.DaysToPD.isna()&
df.DaysToDementia.isna()&
(df.BMI>=18)&(df.BMI<=35)&
(df.HeartAttack_ICD==0)&(df.Stroke_ICD==0)&(df.Anticonvulsants==0)&
(df.RDI<=10)
&(df.Benzodiazapenes==0)&(df.Antiarrhythmics3==0)].reset_index()
print(f'After exclusion criteria (no PD/dementia, BMI 18-35, no HeartAttack/Stroke/Anticonvulsants/Benzos/Antiarrhythmics, RDI<=10): N={len(df)} rows, {df.BDSPPatientID.nunique()} unique patients')

df=df.rename(columns={'DOV':'CreationTime'})
df['CreationTime'] = pd.to_datetime(df.CreationTime)
df = df.drop(columns=['DaysToPD','DaysToDementia','HeartAttack_PreSleepSelfReport','Stroke_PreSleepSelfReport','Hypertension_PreSleepSelfReport','Diabetes_PreSleepSelfReport','Depression_PreSleepSelfReport', 'HeartAttack_ICD', 'Stroke_ICD', 'Anticonvulsants', 'Benzodiazapenes', 'Antiarrhythmics3', 'Age', 'Sex', 'index'])

df2 = pd.read_csv('/data/haoqisun/dataset_HSP/old_metadata/bdsp_psg_master_20231101.csv')
print(f'After loading bdsp_psg_master: N={len(df2)} rows, {df2.BDSPPatientID.nunique()} unique patients')

df2 = df2[df2.StudyType.astype(str).str.lower().str.contains('dia')].reset_index()
print(f'After filtering to diagnostic studies: N={len(df2)} rows, {df2.BDSPPatientID.nunique()} unique patients')

df2['CreationTime'] = pd.to_datetime(df2.CreationTime)
df2 = df2.drop(columns=['StudyType', 'BDSPLastModifiedDTS', 'index'])

df = df2.merge(df, on=['BDSPPatientID', 'CreationTime', 'SessionID'], how='inner', validate='1:1')
print(f'After inner merge (df2 x df): N={len(df)} rows, {df.BDSPPatientID.nunique()} unique patients')
df2['CreationTime'] = df2.CreationTime.dt.strftime('%Y-%m-%d %H:%M:%S')
assert ('sub-'+df.SiteID+df.BDSPPatientID.astype(str)==df.BidsFolder).all()
assert df.BDSPPatientID.nunique()==len(df)

age_bins = [(18,40), (40,60), (60,100)]
sex_bins = ['Male', 'Female']

group_ids = {}
for age, sex in product(age_bins, sex_bins):
    ids = np.where((df.AgeAtVisit>=age[0])&(df.AgeAtVisit<age[1])&(df.SexDSC==sex))[0]
    group_ids[(age,sex)] = ids
    print(f'{age = }, {sex = }: N={len(ids)}')
df.to_csv('mastersheet_v2_for_sciencefair.csv', index=False)
