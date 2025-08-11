from collections import defaultdict
from itertools import product
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
import seaborn as sns
sns.set_style('ticks')
import pwlf
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm


def get_spline(x, y, xtest, nbt=1000, verbose=False, random_seed=None):
    np.random.seed(random_seed)
    ysmooths = []
    for bti in tqdm(range(nbt+1), disable=not verbose):
        if bti==0:
            xbt, ybt = x, y
        else:
            ids = np.random.choice(len(x), len(x), replace=True)
            xbt, ybt = x[ids], y[ids]
        ids2 = np.argsort(xbt)
        spl = UnivariateSpline(xbt[ids2], ybt[ids2])
        ysmooths.append(spl(xtest))
    ysmooths = np.array(ysmooths)
    return ysmooths[0], np.nanpercentile(ysmooths[1:], (2.5,97.5), axis=0)


cols = ['SiteID','BDSPPatientID','SessionID']

#"""
df1=pd.read_csv('S0001_psg_metadata_2025-07-09.csv')

df2=pd.read_csv('ISO_features.csv')
df2b=pd.read_csv('ISO_features_before_paper_submission.csv')
df2 = pd.concat([df2,df2b], axis=0, ignore_index=True)

df2['SiteID']=df2.SID.str.split('-',expand=True)[0]
df2['BDSPPatientID']=df2.SID.str.split('-',expand=True)[1].astype(int)
df2['SessionID']=df2.SID.str.split('-',expand=True)[2].astype(int)
df2=df2.drop(columns='SID')
df1=df1.drop_duplicates(['SiteID','BDSPPatientID','SessionID'],ignore_index=True)

df3 = pd.read_csv('dataset_demog.csv')
df3 = df3.rename(columns={'siteid':'SiteID', 'bdsppatientid':'BDSPPatientID', 'sessionid':'SessionID'})
df3 = df3.drop_duplicates(cols+['dateofbirth'], ignore_index=True)

df=df1.merge(df2,on=cols, how='inner', validate='1:1')
df=df.merge(df3[cols+['dateofbirth']], on=cols, how='inner', validate='1:1')
df['Age'] = (pd.to_datetime(df.CreationTime) - pd.to_datetime(df.dateofbirth)).dt.total_seconds()/86400./365.25
cols_iso = [x for x in df.columns if x.startswith('ISO_')]
df = df.rename(columns={'SexDSC':'Sex'})[cols+['Age', 'Sex']+cols_iso]
print(df)
df.to_csv('dataset_to_plot.csv', index=False)
#"""
#df = pd.read_csv('dataset_to_plot.csv')
cols_iso = [x for x in df.columns if x.startswith('ISO_')]

for x in df.columns:
    if 'power' in x:
        df.loc[:,x]*=100

#age_min, age_max = df.Age.min(), df.Age.max()
age_min, age_max = 18,90


# age & sex histogram

plt.close()
fig = plt.figure(figsize=(8/1.3,3))
ax = fig.add_subplot(121)
ax.hist(df.Age, bins=50)
ax.set_xlabel('Age at Sleep Study')
ax.set_ylabel('Count')
sns.despine()
ax = fig.add_subplot(122)
vals = [(df.Sex=='Female').sum(), (df.Sex=='Male').sum()]
ax.bar([0,1], vals, tick_label=['Female', 'Male'], color=['r','b'], alpha=0.5)
ax.text(0, vals[0], str(vals[0]), ha='center', va='bottom')
ax.text(1, vals[1], str(vals[1]), ha='center', va='bottom')
ax.set_ylabel('Count')
sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig('demo_hist.png', bbox_inches='tight', dpi=300)
#plt.savefig('demo_hist.pdf', bbox_inches='tight', dpi=300)


figure_dir = 'figures'
os.makedirs(figure_dir, exist_ok=True)
xtest = np.arange(age_min, age_max+0.1, 0.1)

random_seed = 2025

metrics = ['bandpower', 'peak_freq', 'peak_relpower']
metrics_txt = ['Band Power (%)', 'Peak Frequency (Hz)', 'Peak Power (%)']
bands = ['fast sigma', 'slow sigma', 'sigma', 'sigma_oof', 'alpha', 'delta']
bands_txt = [r'Fast $\sigma$ (13-15Hz)', r'Slow $\sigma$ (11-13Hz)', r'$\sigma$ (11-15Hz)', r'Aperiodic $\sigma$ (11-15Hz)', r'$\alpha$ (8-12Hz)', r'$\delta$ (1-4Hz)']
colors = {('Female','Frontal'):'lightsalmon', ('Female','Central'):'red', ('Female','Occipital'):'firebrick',
          ('Male','Frontal'):'lightblue', ('Male','Central'):'deepskyblue', ('Male','Occipital'):'royalblue'}
ylims = {'bandpower':[38,55], 'peak_freq':[0.005,0.022], 'peak_relpower':[1.5,2.7]}

"""
save_path = os.path.join(figure_dir, f'bars.png')
plt.close()
fig = plt.figure(figsize=(9.5,6.5))
xpos = np.arange(len(bands))  # the label locations
xlim = [-0.4, len(bands)-1+0.4]
width = 0.1  # the width of the bars
nbt = 1000
np.random.seed(random_seed)
for mi,metric in enumerate(metrics):
    print(metric)
    ax = fig.add_subplot(len(metrics), 1, mi+1)

    vals = defaultdict(list)
    for bti in tqdm(range(nbt+1)):
        if bti==0:
            df_bt = df.copy()
        else:
            df_bt = df.iloc[np.random.choice(len(df),len(df),replace=True)].reset_index(drop=True)

        for s,ch in product(['Female','Male'], ['Frontal', 'Central', 'Occipital']):
            ids = df_bt.Sex==s
            vals[(s,ch)].append([df_bt[f'ISO_{metric}_{b}_{ch}'][ids].median() for b in bands])
    cc = 0
    for k in product(['Female','Male'], ['Frontal', 'Central', 'Occipital']):
        val_ci = np.nanpercentile(np.array(vals[k][1:]), (2.5,97.5), axis=0)
        yerr = np.array([vals[k][0]-val_ci[0],val_ci[1]-vals[k][0]])
        yerr = np.maximum(yerr, 0.0001)
        ax.bar(xpos+(-2.5+cc*1)*width, vals[k][0], yerr=yerr,
               width=width, color=colors[k])
        cc += 1
    # for legend only
    for k in product(['Frontal', 'Central', 'Occipital'], ['Female', 'Male']):
        k2 = (k[1],k[0])
        ax.bar([min(xlim)-10],[10], color=colors[k2], label=f'{k[1]},{k[0]}')
    ax.set_xticks(xpos)
    ax.set_xticklabels(bands_txt)
    ax.set_xlim(xlim)
    if mi==1:
        ax.legend(loc='upper center', ncol=3)
    ax.set_ylabel(metrics_txt[mi])
    ax.set_ylim(ylims[metric])
    ax.yaxis.grid(True)
    sns.despine()
plt.tight_layout()
plt.subplots_adjust(hspace=0.185)
#plt.show()
plt.savefig(save_path, bbox_inches='tight', dpi=300)
"""

np.random.seed(random_seed)

for col in tqdm(cols_iso):
    df_ = df[['Age','Sex',col]].dropna(ignore_index=True)

    #model = pwlf.PiecewiseLinFit(df_.Age, df_[col])
    #res = model.fit(3)
    #print(res)
    #y_smooth1 = model.predict(xtest)
    y_smooth, y_smooth_ci = get_spline(df_.Age.values, df_[col].values, xtest, nbt=1000, random_seed=random_seed)

    save_path = os.path.join(figure_dir, f'age_vs_{col}.png')
    plt.close()
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    ids = df_.Sex=='Female'
    ax.scatter(df_.Age[ids], df_[col][ids], s=20, facecolor='r', edgecolor='none', alpha=0.3, label='Female')
    ids = df_.Sex=='Male'
    ax.scatter(df_.Age[ids], df_[col][ids], s=20, facecolor='b', edgecolor='none', alpha=0.3, label='Male')
    #ax.plot(xtest, y_smooth1, c='r', lw=1.5)
    ax.fill_between(xtest, y_smooth_ci[0], y_smooth_ci[1], color='k', alpha=0.4)
    ax.plot(xtest, y_smooth, c='k', lw=2)
    ax.legend(loc='upper right', scatterpoints=3)
    ax.set_xlim(age_min, age_max)
    if 'peak_freq' in col:
        ax.set_ylim(0, 0.045)
    ax.grid(True)
    ax.set_xlabel('Age (year)')
    ax.set_ylabel(col)
    sns.despine()
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_path, bbox_inches='tight')

    plt.close()
    save_path = os.path.join(figure_dir, f'age_vs_{col}_male_female.png')
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    for color, sex in zip(['r','b'], ['Female', 'Male']):
        ids = df_.Sex==sex
        y_smooth, y_smooth_ci = get_spline(df_.Age.values[ids], df_[col].values[ids], xtest, nbt=1000, random_seed=random_seed+ord(color))
        #ax.scatter(df_.Age, df_[col], s=20, facecolor='k', edgecolor='none', alpha=0.3)
        #ax.plot(xtest, y_smooth1, c='r', lw=1.5)
        ax.fill_between(xtest, y_smooth_ci[0], y_smooth_ci[1], color=color, alpha=0.4)
        ax.plot(xtest, y_smooth, c=color, lw=2, label=sex)
    if 'peak_freq' in col:
        ax.set_ylim(0, 0.045)
    ax.legend(loc='upper right')
    ax.set_xlim(age_min, age_max)
    ax.grid(True)
    ax.set_xlabel('Age (year)')
    ax.set_ylabel(col)
    sns.despine()
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_path, bbox_inches='tight')
