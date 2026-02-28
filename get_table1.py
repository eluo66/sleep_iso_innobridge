import pandas as pd
import numpy as np

df = pd.read_csv('mastersheet_v2_for_sciencefair.csv')
df = df.drop(columns=['BDSPPatientID'])

N = len(df)

# Define continuous variables explicitly; everything else treated as categorical
continuous_vars = [
    'AgeAtVisit', 'BMI', 'AHI', 'AHI_NREM', 'AHI_REM', 'RDI',
    'DaysToLastContact', 'DaysToDeath', 'DaysToDeathMARegistry',
]

# Exclude identifier / constant / datetime columns from Table 1
exclude_vars = ['SiteID', 'CreationTime', 'BidsFolder', 'HasAnnotations', 'HasStaging']

rows = []

for col in df.columns:
    if col in exclude_vars:
        continue

    n_missing = df[col].isna().sum()
    pct_missing = n_missing / N * 100

    if col in continuous_vars:
        mean = df[col].mean()
        sd = df[col].std()
        rows.append({
            'Variable': col,
            'Value': f'{mean:.1f} ({sd:.1f})',
            'Missing N (%)': f'{n_missing} ({pct_missing:.1f}%)',
        })
    else:
        # Categorical: show each level as N (%)
        counts = df[col].value_counts(dropna=True).sort_index()
        first = True
        for val, cnt in counts.items():
            pct = cnt / N * 100
            rows.append({
                'Variable': col if first else '',
                'Value': f'  {val}: {cnt} ({pct:.1f}%)',
                'Missing N (%)': f'{n_missing} ({pct_missing:.1f}%)' if first else '',
            })
            first = False

table1 = pd.DataFrame(rows)
table1.to_excel('table1.xlsx', index=False)

# Print to console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 120)
print(f'Total N = {N}\n')
print(table1.to_string(index=False))
