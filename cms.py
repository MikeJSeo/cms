import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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

# Define visitid: numerical numbering of claims
outpatient_subset['VISIT_ID'] = outpatient_subset.groupby('DESYNPUF_ID').cumcount()+1

merged = pd.merge(benef, outpatient_subset, how = 'left', on = 'DESYNPUF_ID')


# Define age variable
merged['BENE_BIRTH_DT'] = pd.to_datetime(merged['BENE_BIRTH_DT'], format = '%Y%m%d')
merged['AGE'] = (pd.to_datetime("20080101", format = '%Y%m%d') - merged['BENE_BIRTH_DT']) / np.timedelta64(1, 'Y')

# Define death indicator
# merged['Death'] = merged['BENE_DEATH_DT'].notnull().astype(int)

# Redefine binary variable into 0 or 1
binaryvars = ['BENE_SEX_IDENT_CD'] + [col for col in merged if col.startswith('SP_')][1:]
#merged[binaryvars].apply(pd.Series.nunique)
merged[binaryvars] = merged[binaryvars].replace({2:1, 1:0})

# Make dummy variables for multicategory variables
# Maybe include county code (307 unique values) in the future
# Also excluded state code as it didn't seem like an important predictor
# multivars = ['BENE_RACE_CD', 'SP_STATE_CODE']
multivars = ['BENE_RACE_CD']
multicat = pd.get_dummies(merged[multivars], columns = multivars)

columns_to_include = ['DESYNPUF_ID', 'CLM_THRU_DT', 'CLM_PMT_AMT',
                      'BENE_HI_CVRAGE_TOT_MONS', 'BENE_SMI_CVRAGE_TOT_MONS',
                      'BENE_HMO_CVRAGE_TOT_MONS', 'PLAN_CVRG_MOS_NUM',
                      'MEDREIMB_IP', 'BENRES_IP', 'PPPYMT_IP', 
                      'MEDREIMB_OP', 'BENRES_OP', 'PPPYMT_OP', 
                      'MEDREIMB_CAR', 'BENRES_CAR', 'PPPYMT_CAR',
                      'AGE', 'VISIT_ID']
merged = pd.concat([merged[columns_to_include], multicat], axis = 1)
del benef, outpatient, outpatient_subset, multicat

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
     
    time_variant = ['CLM_THRU_DT', 'CLM_PMT_AMT', 'VISIT_ID']

    tf0 = data.copy()
    tf0.drop(time_variant, axis = 1, inplace = True)
    tf0.drop_duplicates('DESYNPUF_ID', inplace = True)
   
    tf = data[(data['CLM_THRU_DT'] >= getTimeFrameDates(k)[0]) & (data['CLM_THRU_DT'] <= getTimeFrameDates(k)[1])]
   
    tf1 = pd.DataFrame({'DESYNPUF_ID': np.unique(tf['DESYNPUF_ID'])})
    tf1['MAX_VISIT_ID'] = np.array(tf.groupby('DESYNPUF_ID')['VISIT_ID'].aggregate('max'))

    aggCLMDates = tf.groupby('DESYNPUF_ID')['CLM_THRU_DT'].aggregate(['min', 'max', 'count'])
    tf1['FIRST_VISIT_FROM_PERIOD_START'] = np.array(((aggCLMDates['min'] -  getTimeFrameDates(k)[0]) / np.timedelta64(1, 'D')))
    tf1['LAST_VISIT_FROM_PERIOD_END'] = np.array(((getTimeFrameDates(k)[1] - aggCLMDates['max']) / np.timedelta64(1, 'D')))
    tf1['INTERVAL_DATES'] = np.array((aggCLMDates['max'] - aggCLMDates['min']) / np.timedelta64(1, 'D'))
    tf1['VISIT_COUNT'] = np.array(aggCLMDates['count'])
    
    aggCLMPMT = tf.groupby('DESYNPUF_ID')['CLM_PMT_AMT'].aggregate(['sum'])
    tf1['TOTAL_PMT'] = np.array(aggCLMPMT['sum'])
    
    tf2 = pd.merge(tf0, tf1, how = 'left')
    del tf, tf0, tf1
    
    tf2['MAX_VISIT_ID'] = tf2['MAX_VISIT_ID'].fillna(0)
    tf2['VISIT_COUNT'] = tf2['VISIT_COUNT'].fillna(0)
    tf2['FIRST_VISIT_FROM_PERIOD_START'] = tf2['FIRST_VISIT_FROM_PERIOD_START'].fillna(0)
    tf2['LAST_VISIT_FROM_PERIOD_END']  = tf2['LAST_VISIT_FROM_PERIOD_END'].fillna(0)
    tf2['INTERVAL_DATES'] = tf2['INTERVAL_DATES'].fillna(0)
    tf2['TOTAL_PMT'] = tf2['TOTAL_PMT'].fillna(0)
    
    # Merge associated test data target
    tf_testID = np.unique(data.loc[(data['CLM_THRU_DT'] >= getTimeFrameDates(k+1)[0]) & (data['CLM_THRU_DT'] < getTimeFrameDates(k+1)[1]), 'DESYNPUF_ID'])
    tf_test = data[(data['CLM_THRU_DT'] >= getTimeFrameDates(k+1)[0]) & (data['CLM_THRU_DT'] < getTimeFrameDates(k+1)[1])]
    
    tf_target = tf_test.groupby('DESYNPUF_ID')['CLM_PMT_AMT'].aggregate('sum')    
    tf_target = pd.DataFrame(tf_target)
    tf_target.reset_index(level=0, inplace=True)
    tf_target['CLAIMED'] = 1
    
    tf_notclaimedID = np.unique(data.loc[~data['DESYNPUF_ID'].isin(tf_testID), 'DESYNPUF_ID'])    
    tf_notclaimed = pd.DataFrame(tf_notclaimedID, columns = ['DESYNPUF_ID'])
    tf_notclaimed['CLM_PMT_AMT'] = 0
    tf_notclaimed['CLAIMED'] = 0
        
    tf_target = pd.concat([tf_target, tf_notclaimed])
    tf_final = pd.merge(tf2, tf_target,on = 'DESYNPUF_ID')  
    del tf2, tf_target, tf_notclaimed

    tf_final['CLM_PMT_AMT'] = tf_final['CLM_PMT_AMT'].astype(float)
    tf_final['CLAIMED'] = tf_final['CLAIMED'].astype(int)

    return tf_final  
        

tr1 = getTimeFramewithFeatures(merged, 1)    
tr2 = getTimeFramewithFeatures(merged, 2)    
tr3 = getTimeFramewithFeatures(merged, 3)    
tr4 = getTimeFramewithFeatures(merged, 4)
tr5 = getTimeFramewithFeatures(merged, 5)

tr5['CLAIMED'] = np.nan
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

train_all = train.drop(['DESYNPUF_ID', 'CLM_PMT_AMT', 'CLAIMED'], axis = 1)
train_all_label = train['CLAIMED']
dtrain_all = lgb.Dataset(train_all, label = train_all_label)

train_claimed = train[train['CLAIMED'] == 1]
train_claimed_label = train_claimed['CLM_PMT_AMT']
train_claimed = train_claimed.drop(['DESYNPUF_ID', 'CLM_PMT_AMT', 'CLAIMED'], axis = 1)
dtrain_claimed = lgb.Dataset(train_claimed, label = train_claimed_label)


lgb_model1 = lgb.train(params_classification, dtrain_all, 
                      num_boost_round=5000,
                      valid_sets= dtrain_all,
                      early_stopping_rounds=100)

test_all = test.drop(['DESYNPUF_ID', 'CLM_PMT_AMT', 'CLAIMED'], axis = 1)
pr_lgb = lgb_model1.predict(test_all, num_iteration = lgb_model1.best_iteration)

lgb_model2 = lgb.train(params_regression, dtrain_claimed, 
                      num_boost_round=5000,
                      valid_sets= dtrain_claimed,
                      early_stopping_rounds=100)
amt_lgb = lgb_model2.predict(test_all, num_iteration = lgb_model2.best_iteration)

amt_pred = pd.Series(pr_lgb * amt_lgb)
amt_true = getTimeFramewithFeatures(merged, 5)['CLM_PMT_AMT']

def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)
rmse(amt_pred, amt_true)
#778.80118
#632.73565
#595.05552

pd.set_option('min_rows', 100)
pd.concat([amt_pred, amt_true], axis = 1)


#### plotting importance
def plot_importance(model):
    df_feature_importance = (
        pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(),
            })
        .sort_values('importance', ascending=False)
    )
    
    plt.figure(figsize=(14,10))
    sns.barplot(x="importance", y="feature", data= df_feature_importance)
    plt.title('LightGBM Features')
    plt.tight_layout()

plot_importance(lgb_model1)    
plot_importance(lgb_model2)

#Top 10
#MEDREIMB_OP - Outpatient Institutional annual Medicare reimbursement amount
#TOTAL_PMT
#MAX_VISIT_ID
#BENRES_OP - Outpatient Institutional annual beneficiary responsibility amount
#MEDREIMB_CAR - Carrier annual Medicare reimbursement amount
#BENRES_CAR - Carrier annual beneficiary responsibility amount
#INTERVAL_DATES
#AGE
#LAST_VISIT_FROM_PERIOD_END
#VISIT_COUNT
