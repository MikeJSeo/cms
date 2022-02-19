import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/mike/Desktop/")
benef = pd.read_csv("beneficiary summary.csv")

# EDA
len(set(benef['DESYNPUF_ID']))
benef['BENE_SEX_IDENT_CD'].value_counts()/ benef.shape[0]
benef['BENE_RACE_CD'].value_counts()/ benef.shape[0]

# Defining age variable
benef['BENE_BIRTH_DT'] = pd.to_datetime(benef['BENE_BIRTH_DT'], format = '%Y%m%d')
threshold = pd.to_datetime("20080101", format = '%Y%m%d')
benef['age'] = (threshold - benef['BENE_BIRTH_DT']) / np.timedelta64(1, 'Y')

# merge inpatient, outpatient claims dataset
outpatient = pd.read_csv('outpatient.csv')
inpatient = pd.read_csv('inpatient.csv')

outpatient_small = outpatient[['DESYNPUF_ID', 'CLM_THRU_DT', 'CLM_PMT_AMT']]

# drop patients with null CLM_THRU_DT and CLM_PMT_AMT
outpatient_small = outpatient_small.dropna(subset = ['CLM_THRU_DT', 'CLM_PMT_AMT'])

outpatient_small[all(pd.notnull(outpatient_small[['CLM_THRU_DT', 'CLM_PMT_AMT']]))]
outpatient_small['CLM_THRU_DT'] = outpatient_small['CLM_THRU_DT'].apply(np.int)
outpatient_small['CLM_THRU_DT'] =  pd.to_datetime(outpatient_small['CLM_THRU_DT'], format = '%Y%m%d')
outpatient_small['claim_date'] = (outpatient_small['CLM_THRU_DT'] - threshold)/ np.timedelta64(1, 'Y')

merged = pd.merge(benef, outpatient_small, how = 'left')


d = merged['DESYNPUF_ID'].value_counts()

import matplotlib.pyplot as plt
n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')


