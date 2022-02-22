import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from datetime import timedelta


# File is available in cms.gov website:
# https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DESample01
# Relevant files are: DE1.0 Sample 1 2008 Beneficiary Summary File, DE1.0 Sample 1 2008-2010 Inpatient Claims, 
# and DE1.0 Sample 1 2008-2010 Outpatient Claims 

os.chdir("C:/Users/mike/Desktop/")
benef = pd.read_csv("benef.csv")

# merge outpatient claims dataset
outpatient = pd.read_csv('outpatient.csv')
#inpatient = pd.read_csv('inpatient.csv')

columns_from_outpatient = ['DESYNPUF_ID', 'CLM_THRU_DT', 'CLM_PMT_AMT']
outpatient_subset = outpatient[columns_from_outpatient]

# change claim-through-date to datetime format
outpatient_subset = outpatient_subset.dropna(subset = ['CLM_THRU_DT'])
outpatient_subset['CLM_THRU_DT'] = outpatient_subset['CLM_THRU_DT'].apply(np.int)
outpatient_subset['CLM_THRU_DT'] = pd.to_datetime(outpatient_subset['CLM_THRU_DT'], format = '%Y%m%d')

merged = pd.merge(benef, outpatient_subset, how = 'left', on = 'DESYNPUF_ID')

# Define age variable
merged['BENE_BIRTH_DT'] = pd.to_datetime(merged['BENE_BIRTH_DT'], format = '%Y%m%d')
threshold = pd.to_datetime("20080101", format = '%Y%m%d')
merged['Age'] = (threshold - merged['BENE_BIRTH_DT']) / np.timedelta64(1, 'Y')

# Define death indicator
# merged['Death'] = merged['BENE_DEATH_DT'].notnull().astype(int)

# Redefine binary variable into 0 or 1
dummyvars = ['BENE_SEX_IDENT_CD'] + [col for col in merged if col.startswith('SP_')][1:]
#merged[dummyvars].apply(pd.Series.nunique)
merged[dummyvars] = merged[dummyvars].replace({2:1, 1:0})

# Make dummy variables for multicategory variables
# Maybe include county code (307 unique values) in the future
multivars = ['BENE_RACE_CD', 'SP_STATE_CODE']
multicat = pd.get_dummies(merged[multivars], columns = multivars)

# Select variables to include
columns_to_include = ['DESYNPUF_ID', 'CLM_THRU_DT',
                      'CLM_PMT_AMT', 'Age',
                      'BENE_HI_CVRAGE_TOT_MONS', 'BENE_SMI_CVRAGE_TOT_MONS',
                      'BENE_HMO_CVRAGE_TOT_MONS', 'PLAN_CVRG_MOS_NUM',
                      'MEDREIMB_IP', 'BENRES_IP', 'PPPYMT_IP', 
                      'MEDREIMB_OP', 'BENRES_OP', 'PPPYMT_OP', 
                      'MEDREIMB_CAR', 'BENRES_CAR', 'PPPYMT_CAR']
merged = pd.concat([merged[columns_to_include], multicat], axis = 1)



###################################
## Setup timeFrame is 6 month to predict next 6 month
## Ultimate goal is to predict claim amount for the last time frame: 07/01/2010 to 12/31/2010

def getTimeFrameDates(k):
    if k % 2 == 1:
        startdate = pd.to_datetime('01/01/' + str(2008 + (k-1)//2))
        enddate = pd.to_datetime('07/01/' + str(2008 + (k-1)//2))
    else:
        startdate = pd.to_datetime('07/01/' + str(2008 + (k-1)//2))
        enddate = pd.to_datetime('12/31/' + str(2008 + (k-1)//2))
    return [startdate, enddate]


## Building training set for each time frame

def getTimeFramewithFeatures(data, k):
           
    tf = data[(data['CLM_THRU_DT'] >= getTimeFrameDates(k)[0]) & (data['CLM_THRU_DT'] <= getTimeFrameDates(k)[1])]
    
    tf_testID = np.unique(data.loc[(data['CLM_THRU_DT'] >= getTimeFrameDates(k+1)[0]) & (data['CLM_THRU_DT'] < getTimeFrameDates(k+1)[1]), 'DESYNPUF_ID'])
    tf_test = data[(data['CLM_THRU_DT'] >= getTimeFrameDates(k+1)[0]) & (data['CLM_THRU_DT'] < getTimeFrameDates(k+1)[1])]
    
    tf_target = tf_test.groupby('DESYNPUF_ID')['CLM_PMT_AMT'].aggregate('sum')    
    tf_target = pd.DataFrame(tf_target)
    tf_target.reset_index(level=0, inplace=True)
    tf_target['claimed'] = 1
    
    tf_notclaimedID = np.unique(data.loc[~data['DESYNPUF_ID'].isin(tf_testID), 'DESYNPUF_ID'])    
    tf_notclaimed = pd.DataFrame(tf_notclaimedID, columns = ['DESYNPUF_ID'])
    tf_notclaimed['CLM_PMT_AMT'] = 0
    tf_notclaimed['claimed'] = 0
        
    tf_target = pd.concat([tf_target, tf_notclaimed])
    tf2 = data.copy()
    tf2.drop(['CLM_THRU_DT', 'CLM_PMT_AMT'], axis = 1, inplace = True)
    tf2.drop_duplicates('DESYNPUF_ID', inplace = True)    
    tf_final = pd.merge(tf2, tf_target,on = 'DESYNPUF_ID')

    tf_final['CLM_PMT_AMT'] = tf_final['CLM_PMT_AMT'].astype(float)
    tf_final['claimed'] = tf_final['claimed'].astype(int)

    return tf_final  
        

tr1 = getTimeFramewithFeatures(merged, 1)    
tr2 = getTimeFramewithFeatures(merged, 2)    
tr3 = getTimeFramewithFeatures(merged, 3)    
tr4 = getTimeFramewithFeatures(merged, 4)
tr5 = getTimeFramewithFeatures(merged, 5)

tr5['claimed'] = np.nan
tr5['CLM_PMT_AMT'] = np.nan


train_all = pd.concat([tr1, tr2, tr3, tr4, tr5])
train = train_all[~train_all['CLM_PMT_AMT'].isnull()]
test = train_all[train_all['CLM_PMT_AMT'].isnull()]


params_classification = {
    "objective" : "binary",
    "metric" : "binary_logloss",
    "max_bin": 256,
    "num_leaves" : 15,
    "learning_rate" : 0.01,
    "bagging_fraction" : 0.9,
    "feature_fraction" : 0.8,
    "bagging_seed" : 42,
    "verbosity" : -1,
    "seed": 42
}

params_regression = {
    "objective" : "regression",
    "metric" : "rmse",
    "max_bin": 256,
    "num_leaves" : 9,
    "learning_rate" : 0.01,
    "bagging_fraction" : 0.9,
    "feature_fraction" : 0.8,
    "bagging_seed" : 42,
    "verbosity" : -1,
    "seed": 42
}


train_all = train.drop(['DESYNPUF_ID', 'CLM_PMT_AMT', 'claimed'], axis = 1)
train_all_label = train['claimed']
dtrain_all = lgb.Dataset(train_all, label = train_all_label)

train_claimed = train[train['claimed'] == 1]
train_claimed_label = train_claimed['CLM_PMT_AMT']
train_claimed = train_claimed.drop(['DESYNPUF_ID', 'CLM_PMT_AMT', 'claimed'], axis = 1)
dtrain_claimed = lgb.Dataset(train_claimed, label = train_claimed_label)


lgb_model1 = lgb.train(params_classification, dtrain_all, 
                      num_boost_round=5000,
                      valid_sets= dtrain_all,
                      early_stopping_rounds=100,
                      verbose_eval=500)

test_all = test.drop(['DESYNPUF_ID', 'CLM_PMT_AMT', 'claimed'], axis = 1)
pr_lgb = lgb_model1.predict(test_all, num_iteration = lgb_model1.best_iteration)

lgb_model2 = lgb.train(params_regression, dtrain_claimed, 
                      num_boost_round=5000,
                      valid_sets= dtrain_claimed,
                      early_stopping_rounds=100,
                      verbose_eval=500)
amt_lgb = lgb_model2.predict(test_all, num_iteration = lgb_model2.best_iteration)

final_amt = pd.Series(pr_lgb * amt_lgb)

real_final_amt = getTimeFramewithFeatures(merged, 5)['CLM_PMT_AMT']

def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)
rmse(real_final_amt, final_amt)
#778.80118

pd.set_option('min_rows', 100)
pd.concat([real_final_amt, final_amt], axis = 1)



