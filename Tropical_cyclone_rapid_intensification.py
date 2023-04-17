#!/usr/bin/env python
# coding: utf-8

# In[186]:


import keras
import pickle
import time
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, r2_score, confusion_matrix, ConfusionMatrixDisplay, recall_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn.metrics import roc_curve, auc
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import NearMiss 
from imblearn.over_sampling import SMOTE


# In[187]:


df = pd.DataFrame(pd.read_csv("global.csv"))
df = df.drop(['date', 'hour'], axis = 1)
#df


# In[188]:


feat = ['vs0', 'PSLV_v2', 'PSLV_v3', 'PSLV_v4', 'PSLV_v5', 'PSLV_v6', 'PSLV_v7',
        'PSLV_v8', 'PSLV_v9', 'PSLV_v10', 'PSLV_v11', 'PSLV_v12', 'PSLV_v13',
        'PSLV_v14', 'PSLV_v15', 'PSLV_v16', 'PSLV_v17', 'PSLV_v18', 'PSLV_v19',
        'MTPW_v2', 'MTPW_v3', 'MTPW_v4', 'MTPW_v5', 'MTPW_v6', 'MTPW_v7',
        'MTPW_v8', 'MTPW_v9', 'MTPW_v10', 'MTPW_v11', 'MTPW_v12', 'MTPW_v13',
        'MTPW_v14', 'MTPW_v15', 'MTPW_v16', 'MTPW_v17', 'MTPW_v18', 'MTPW_v19',
        'MTPW_v20', 'MTPW_v21', 'MTPW_v22', 'IR00_v2', 'IR00_v3', 'IR00_v4',
        'IR00_v5', 'IR00_v6', 'IR00_v7', 'IR00_v8', 'IR00_v9', 'IR00_v10',
        'IR00_v11', 'IR00_v12', 'IR00_v13', 'IR00_v14', 'IR00_v15', 'IR00_v16',
        'IR00_v17', 'IR00_v18', 'IR00_v19', 'IR00_v20', 'IR00_v21', 'CSST_t24',
        'CD20_t24', 'CD26_t24', 'COHC_t24', 'DTL_t24', 'RSST_t24', 'U200_t24',
        'U20C_t24', 'V20C_t24', 'E000_t24', 'EPOS_t24', 'ENEG_t24', 'EPSS_t24',
        'ENSS_t24', 'RHLO_t24', 'RHMD_t24', 'RHHI_t24', 'Z850_t24', 'D200_t24',
        'REFC_t24', 'PEFC_t24', 'T000_t24', 'R000_t24', 'Z000_t24', 'TLAT_t24',
        'TLON_t24', 'TWAC_t24', 'TWXC_t24', 'G150_t24', 'G200_t24', 'G250_t24',
        'V000_t24', 'V850_t24', 'V500_t24', 'V300_t24', 'TGRD_t24', 'TADV_t24',
        'PENC_t24', 'SHDC_t24', 'SDDC_t24', 'SHGC_t24', 'DIVC_t24', 'T150_t24',
        'T200_t24', 'T250_t24', 'SHRD_t24', 'SHTD_t24', 'SHRS_t24', 'SHTS_t24',
        'SHRG_t24', 'PENV_t24', 'VMPI_t24', 'VVAV_t24', 'VMFX_t24', 'VVAC_t24',
        'HE07_t24', 'HE05_t24', 'O500_t24', 'O700_t24', 'CFLX_t24', 'DELV-12', 'dvs24']

features = ['COHC_t24', 'DELV-12', 'CD26_t24', 'VMPI_t24', 'CD20_t24', 'VMFX_t24',
            'vs0', 'EPSS_t24', 'VVAV_t24', 'CFLX_t24','TGRD_t24', 'RSST_t24', 
            'VVAC_t24', 'EPOS_t24', 'T000_t24', 'SHRG_t24', 'SHRD_t24', 'ENSS_t24',
            'SHGC_t24', 'T150_t24', 'dvs24']

feat_IO = ['vs0','PSLV_v5','PSLV_v6','PSLV_v7','PSLV_v8',
           'PSLV_v10','PSLV_v11','PSLV_v12','PSLV_v15','PSLV_v17',
           'PSLV_v19','MTPW_v2','MTPW_v3','MTPW_v5','MTPW_v7',
           'MTPW_v8','MTPW_v9','MTPW_v10','MTPW_v11','MTPW_v12',
           'MTPW_v13','MTPW_v16','MTPW_v18','MTPW_v19','MTPW_v20',
           'IR00_v3','IR00_v6','IR00_v8','IR00_v9','IR00_v10',
           'IR00_v15','IR00_v16','IR00_v17','IR00_v18','IR00_v20',
           'CD26_t24','dvs24']

feat_IO_avgprecision = ['vs0','PSLV_v4','PSLV_v6','PSLV_v9','PSLV_v11','PSLV_v12',
                        'PSLV_v14','PSLV_v16','PSLV_v18','PSLV_v19','MTPW_v3','MTPW_v4',
                        'MTPW_v5','MTPW_v8','MTPW_v9','MTPW_v10','MTPW_v11','MTPW_v12',
                        'MTPW_v15','MTPW_v19','MTPW_v21','IR00_v2','IR00_v4','IR00_v13',
                        'IR00_v15','IR00_v18','CSST_t24','DTL_t24','RSST_t24','U200_t24',
                        'U20C_t24','V20C_t24','ENEG_t24','EPSS_t24','REFC_t24','T000_t24',
                        'TLAT_t24','TLON_t24','TWAC_t24','TWXC_t24','G200_t24','G250_t24',
                        'V000_t24','V850_t24','V500_t24','V300_t24','TGRD_t24','SHDC_t24',
                        'SDDC_t24','SHGC_t24','DIVC_t24','SHRS_t24','VVAV_t24','O500_t24',
                        'DELV-12','dvs24']

atlantic_sfs_20 = ['vs0','PSLV_v4','PSLV_v9','PSLV_v12','PSLV_v16',
                   'MTPW_v9','MTPW_v18','IR00_v18','CD20_t24','DTL_t24',
                   'U20C_t24','PEFC_t24','Z000_t24','TLON_t24','G200_t24',
                   'V850_t24','V300_t24','VMPI_t24','O700_t24','DELV-12','dvs24']

indian_sfs_18 = ['vs0','MTPW_v19','IR00_v2','IR00_v15','CSST_t24',
                   'DTL_t24','RSST_t24','V20C_t24','ENEG_t24','EPSS_t24',
                   'TLON_t24','TWXC_t24','V850_t24','V300_t24','SDDC_t24',
                   'SHGC_t24','SHRS_t24','DELV-12','dvs24']

indian_sfs_55 = ['vs0','PSLV_v4','PSLV_v6','PSLV_v9','PSLV_v11','PSLV_v12','PSLV_v14','PSLV_v16',
                 'PSLV_v18','PSLV_v19','MTPW_v3','MTPW_v4','MTPW_v5','MTPW_v8','MTPW_v9','MTPW_v10',
                 'MTPW_v11','MTPW_v12','MTPW_v15','MTPW_v19','MTPW_v21','IR00_v2','IR00_v4','IR00_v13',
                 'IR00_v15','IR00_v18','CSST_t24','DTL_t24','RSST_t24','U200_t24','U20C_t24','V20C_t24',
                 'ENEG_t24','EPSS_t24','REFC_t24','T000_t24','TLAT_t24','TLON_t24','TWAC_t24','TWXC_t24',
                 'G200_t24','G250_t24','V000_t24','V850_t24','V500_t24','V300_t24','TGRD_t24','SHDC_t24',
                 'SDDC_t24','SHGC_t24','DIVC_t24','SHRS_t24','VVAV_t24','O500_t24','DELV-12','dvs24']

atlantic_sfs_new = ['vs0','PSLV_v17','MTPW_v11','IR00_v4','IR00_v18',
                     'CSST_t24','CD26_t24','COHC_t24','DTL_t24','RSST_t24',
                     'RHLO_t24','Z850_t24','TLAT_t24','TWAC_t24','G150_t24',
                     'V000_t24','V500_t24','V300_t24','PENC_t24','SHGC_t24',
                     'T250_t24','SHRS_t24','VVAV_t24','VVAC_t24','DELV-12','dvs24']

#atlantic_rf = ['DELV-12',     'ENSS_t24',    'ENEG_t24',    'vs0',         'DTL_t24',    
#            'VMPI_t24',    'RSST_t24',    'U20C_t24',    'TWXC_t24',    'SHGC_t24',  
#            'SHTS_t24',    'SHRG_t24',   'CD20_t24' ,   'SHRD_t24',   'SHDC_t24',    
#            'TADV_t24',    'EPSS_t24',    'COHC_t24',    'RHLO_t24',    'T150_t24', 'dvs24']

atlantic_rf = ['vs0',        'DTL_t24',    'DELV-12',    'CFLX_t24',    
               'SHGC_t24',   'SHDC_t24',   'TWXC_t24',    'TWAC_t24',    'V500_t24',   
               'TLON_t24',    'CD20_t24',   'VMPI_t24',  'IR00_v3',     'V000_t24',   'TLAT_t24',
               'CD26_t24',   'VVAC_t24',    'SHRG_t24',   'EPOS_t24',   'MTPW_v3',    'dvs24']

indian_10_rf =  ['vs0',        'DTL_t24',    'DELV-12',     'MTPW_v18',    'SHGC_t24',    
                 'COHC_t24',   'SHDC_t24',   'TWXC_t24',    'V300_t24',  'CFLX_t24', 'dvs24']

atlantic_22_rf = ['vs0',         'DTL_t24',     'DELV-12',     'CFLX_t24',    'SHGC_t24',    
                  'SHDC_t24',    'TWXC_t24',    'TWAC_t24',    'V500_t24',    'TLON_t24',    'CD20_t24',    
                  'VMPI_t24',    'IR00_v3',     'V000_t24',    'TLAT_t24',    'CD26_t24',    'VVAC_t24',    
                  'SHRG_t24',   'EPOS_t24',    'MTPW_v3',    'COHC_t24',    'V300_t24',      'dvs24']

west_13_rf = ['DELV-12',      'vs0',         'COHC_t24',    'DTL_t24',     'TLON_t24',    
             'TWXC_t24',    'VMPI_t24',    'CSST_t24',    'V500_t24',    'CD20_t24',    
             'SHGC_t24',    'TWAC_t24',    'RSST_t24',    'dvs24']

east_36_rf = ['DELV-12',     'vs0',         'ENEG_t24',    'DTL_t24',     'CFLX_t24',    'VVAC_t24',    
             'TWXC_t24',    'SHGC_t24',    'SHDC_t24',    'IR00_v5',    'VMPI_t24',    'ENSS_t24',    
             'U200_t24',    'TWAC_t24',    'SHRG_t24',    'TLAT_t24',    'SHRD_t24',    'PSLV_v6',     
             'V300_t24',   'U20C_t24',    'RSST_t24',    'VMFX_t24',    'D200_t24',    'PSLV_v8',    
             'CD26_t24',   'V850_t24',    'V500_t24',    'V000_t24',    'E000_t24',   'PSLV_v5',     
             'PSLV_v16',   'TLON_t24',    'MTPW_v11',   'MTPW_v11',    'CD20_t24',    'Z850_t24', 'dvs24']


# In[1019]:


common_elements = set(east_36_rf).intersection(atlantic_22_rf, west_13_rf, indian_10_rf)

print(common_elements)


# In[333]:


#df1 = pd.DataFrame(df, columns = feat)                     # Global data
#df1 = pd.DataFrame(df.iloc[44660:54960], columns = atlantic_22_rf)   # Atlantic ocean (upto 2016) 
#df1 = pd.DataFrame(df.iloc[26616:44204], columns = feat)   # Western pacific (upto 2016)
df1 = pd.DataFrame(df.iloc[13399:15547], columns = indian_10_rf)   # Indian Ocean (upto 2014)
#df1 = pd.DataFrame(df.iloc[13399:15729], columns = indian_10_rf)   # Indian Ocean (upto 2016)
#df1 = pd.DataFrame(df.iloc[: 13062], columns = east_36_rf)       # Eastern Pacific (upto 2016) 

#df1 = df1.dropna(axis = 1)
#df1 = df1.drop(df1.index[49337:55370])
df1


# In[334]:


#df3 = pd.DataFrame(df.iloc[44204:44662], columns = feat)   # WP 2017
#df3 = pd.DataFrame(df.iloc[54960:55370], columns = atlantic_22_rf)   # AL 2017
df3 = pd.DataFrame(df.iloc[15547:15778], columns = indian_10_rf)   # IO 2015-2017
#df3 = pd.DataFrame(df.iloc[13062:13397], columns = east_36_rf)   # EP 2017

#df3 = pd.DataFrame(pd.read_csv("NOAA_operational_vars_global_w_dvs24.csv"))  # operational AL 2010-16
#df3 = pd.DataFrame(df3, columns = feat)
#df3 = df3.dropna(axis = 1)

#df3 = pd.DataFrame(pd.read_csv("atlantic_2019-20.csv"))  # operational AL 2019-20
#df3 = pd.DataFrame(df3, columns = feat)
df3


# FEATURE SELECTION BASED ON RANDOM FOREST

# In[294]:


start = time.time()

x = df1.drop(['dvs24'], axis = 1)
y = df1['dvs24']

#tree = DecisionTreeRegressor()
#tree.fit(x, y)

rf = RandomForestRegressor(n_estimators = 500, random_state = 1)
rf.fit(x, y)

feature_list = list(x.columns)
feature_imp = pd.Series(rf.feature_importances_, index = feature_list).sort_values(ascending = False)
print(feature_imp[:])

end = time.time()
print('Time:', (end-start)/60)


# In[133]:


print(feature_imp[:22])


# In[130]:


for i in range(1, 122):
    #df_1 = pd.DataFrame(df.iloc[44660:54960], columns = feature_imp.index[: i])
    #df_3 = pd.DataFrame(df.iloc[54960:55370], columns = feature_imp.index[: i])
    #df_1 = pd.DataFrame(df.iloc[26616:44204], columns = feature_imp.index[: i])
    #df_3 = pd.DataFrame(df.iloc[44204:44662], columns = feature_imp.index[: i])
    #df_1 = pd.DataFrame(df.iloc[13399:15547], columns = feature_imp.index[: i])
    #df_3 = pd.DataFrame(df.iloc[15547:15778], columns = feature_imp.index[: i])
    df_1 = pd.DataFrame(df.iloc[: 13062], columns = feature_imp.index[: i])       
    df_3 = pd.DataFrame(df.iloc[13062:13397], columns = feature_imp.index[: i])
    print(df_1)


# RUNNING SVM CLASSIFICATION USING FEATURES FROM RANDOM FOREST RANKING BY ADDING ONE FEATURE ITERATIVELY

# In[296]:


start = time.time()

arr_pod, arr_far, arr_f1, arr_pss, arr_gss, arr_hss = [], [], [], [], [], []

for i in range(1, 122):
    df_1 = pd.DataFrame(df.iloc[44660:54960], columns = feature_imp.index[: i])  # Atlantic 
    df_3 = pd.DataFrame(df.iloc[54960:55370], columns = feature_imp.index[: i])
    #df_1 = pd.DataFrame(df.iloc[26616:44204], columns = feature_imp.index[: i])  # Western
    #df_3 = pd.DataFrame(df.iloc[44204:44662], columns = feature_imp.index[: i])
    #df_1 = pd.DataFrame(df.iloc[13399:15547], columns = feature_imp.index[: i])  # Indian
    #df_3 = pd.DataFrame(df.iloc[15547:15778], columns = feature_imp.index[: i])
    #df_1 = pd.DataFrame(df.iloc[: 13062], columns = feature_imp.index[: i])       # Eastern
    #df_3 = pd.DataFrame(df.iloc[13062:13397], columns = feature_imp.index[: i])
    
    cols_train, cols_test = list(df_1)[:], list(df_3)[:]   
    dftrain, dftest = df_1[cols_train].astype(float), df_3[cols_test].astype(float)
    
    scaler = StandardScaler()
    dftrain_scaled, dftest_scaled = scaler.fit_transform(dftrain), scaler.fit_transform(dftest)
    x_train, y_train, x_test, y_test = dftrain_scaled[:], np.array(df1['dvs24']), dftest_scaled[:], np.array(df3['dvs24'])
    
    # Convert y_train and y_test to binary labels
    for j in range(len(y_train)):
        if y_train[j] >= 30:
            y_train[j] = 1
        else:
            y_train[j] = 0
            
    for k in range(len(y_test)):
        if y_test[k] >= 30:
            y_test[k] = 1
        else:
            y_test[k] = 0
    
    # Apply SMOTE to handle imbalanced data
    smote = SMOTE(random_state = 42, sampling_strategy = {0: 9768, 1: 2442}, k_neighbors = 10)                        
    xsample, ysample = smote.fit_resample(x_train, y_train)
    
    # Train the SVC classifier
    svc = SVC(kernel = 'rbf', gamma = 0.005, C = 15, verbose = 2)
    classifier = svc.fit(xsample, ysample)
    
    # Make predictions and evaluate the classifier
    ypredict = classifier.predict(x_test)  
    print(i, '\n', classification_report(y_test, ypredict))
    arr_f1.append(f1_score(y_test, ypredict))

    cm = confusion_matrix(y_test, ypredict)
    tp, tn = cm[1][1], cm[0][0]
    fp, fn = cm[0][1], cm[1][0]
    pod = tp/(tp+fn)
    far = fp/(tp+fp)
    pss = pod-(fp/(fp+tn))
    acc = (tp+tn)/(tp+tn+fp+fn)
    hr = (tp+fn)*(tp+fp)/(tp+fp+tn+fn)
    gss = (tp-hr)/(tp+fn+fp-hr)
    kp = (2*((tp*tn)-(fn*fp)))/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)) 
    exp = (1/len(x_test))*((tp+fn)*(tp+fp)+(tn+fn)*(tn+fp))
    hss = ((tp+tn)-exp)/(len(x_test)-exp)

    arr_pod.append(pod)
    arr_far.append(far)
    arr_hss.append(hss)
    arr_pss.append(pss)
    arr_gss.append(gss)
    
end = time.time()
print('TIME ELSPASED:', (end - start)/60, 'min')    


# In[297]:


no_samples = np.arange(1, 122, 1)
xticks = [1, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121]
metric_arr = [arr_pod, arr_far, arr_f1, arr_pss, arr_gss, arr_hss]
metric_labels = ['POD', 'FAR', 'f-1 Score', 'PSS', 'GSS', 'HSS']

fig, ax = plt.subplots(2, 3, figsize = (15, 8), sharex = True)
ax = ax.ravel()
warnings.filterwarnings("ignore")
print('\n')

for i in range(len(metric_arr)):
    fig.suptitle('Performance metrics vs number of features based on random forest ranking', fontsize = 18)
    ax[i].plot(no_samples, metric_arr[i], 'o-', linewidth = 1, markersize = 2.5)
    ax[i].set_ylabel(metric_labels[i], fontsize = 14, labelpad = 8)
    ax[i].tick_params(direction = 'in')
    
    if i >= 3:
        ax[i].set_xlabel('Number of features', fontsize = 14, labelpad = 8)
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticks, fontsize = 12)
    else:
        ax[i].tick_params(direction = 'in')
        
    ax[i].set_yticklabels(['%.2f' % y for y in ax[i].get_yticks()], fontsize = 12)
    
plt.tight_layout()   
#plt.savefig('Westpacific_metrics_randomforest.jpg', dpi = 1200)
plt.show()


# In[315]:


plt.figure(figsize = (8, 6))
plt.plot(no_samples, arr_pod, 'o-', label = 'POD', color= 'blue', linewidth = 1, markersize = 3)
plt.plot(no_samples, arr_far, 'o-', label = 'FAR', color = 'red', linewidth = 1, markersize = 3)
plt.xlabel('Number of features', fontsize = 16, labelpad = 10)
plt.axvline(x=22, color='black', linestyle='--', linewidth = 0.5) # Add vertical line at x=5
#plt.ylabel('Skill Score', fontsize = 16, labelpad = 10)
plt.xticks(xticks, fontsize = 14)
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.ylim(-0.05, 1.05)
plt.tick_params(direction = 'in', length = 5, width = 1, labelsize = 14, pad = 8)
#bold_text = plt.text(0.10, 0.9, '(b)', transform = plt.gca().transAxes, fontsize = 15)
#bold_text.set_weight(550)
plt.legend(fontsize = 12)
#plt.savefig('Indian new metric plot.jpg', dpi = 1200)
plt.show()


# In[327]:


arr_pod[14]


# In[328]:


arr_far[14]


# In[332]:


arr_f1[14]


# In[264]:


plt.figure(figsize = (8, 6))
plt.plot(no_samples, arr_f1, 'o-', label = 'F1 score', linewidth = 1, markersize = 3)
plt.xlabel('Number of features', fontsize = 16, labelpad = 10)
#plt.ylabel('F1 score', fontsize = 16, labelpad = 10)
plt.xticks(xticks, fontsize = 14)
plt.yticks(np.arange(0.0, 0.8, 0.1), [])
plt.ylim(-0.05, 0.75)
plt.tick_params(direction = 'in', length = 5, width = 1, labelsize = 14, pad = 8)
bold_text = plt.text(0.05, 0.9, '(b)', transform = plt.gca().transAxes, fontsize = 15)
bold_text.set_weight(550)
#plt.savefig('Indian new F1 plot.jpg', dpi = 1200)
plt.show()


# In[95]:


arr_pod = np.array(arr_pod)
arr_far = np.array(arr_far)


# In[51]:


import ast

df_features = pd.DataFrame(pd.read_csv("sfs_indian_basin.csv"))
dfeatures = df_features['feature_idx']

arrr_pod, arrr_far, arrr_f1, arrr_pss, arrr_gss, arrr_hss = [], [], [], [], [], []


for i in range(len(dfeatures)):
    index = ast.literal_eval(dfeatures[i])
    converted = []
    for j in range(len(index)):
        converted.append(int(index[j]))

    final_features = []
    for k in range(len(converted)):
        final_features.append(feat[converted[k]])
        
    #df_tr = pd.DataFrame(df.iloc[44660:54960], columns = final_features)
    #df_te = pd.DataFrame(df.iloc[54960:55370], columns = final_features)
    df_tr = pd.DataFrame(df.iloc[13399:15547], columns = final_features)
    df_te = pd.DataFrame(df.iloc[15547:15778], columns = final_features)
    
    cols_tr, cols_te = list(df_tr), list(df_te)   
    dftr, dfte = df_tr[cols_tr].astype(float), df_te[cols_te].astype(float)
    
    scaler = StandardScaler()
    dftr_scaled, dfte_scaled = scaler.fit_transform(dftr), scaler.fit_transform(dfte)
    x_tr, y_tr, x_te, y_te = dftr_scaled[:], np.array(df1['dvs24']), dfte_scaled[:], np.array(df3['dvs24'])
    
    # Convert y_train and y_test to binary labels
    for l in range(len(y_tr)):
        if y_tr[l] >= 30:
            y_tr[l] = 1
        else:
            y_tr[l] = 0
            
    for m in range(len(y_te)):
        if y_te[m] >= 30:
            y_te[m] = 1
        else:
            y_te[m] = 0
    
    smote2 = SMOTE(random_state = 42, sampling_strategy = {0: 2020, 1: 2020}, k_neighbors = 10)                        
    xsample2, ysample2 = smote2.fit_resample(x_tr, y_tr)
    
    svc2 = SVC(kernel = 'rbf', gamma = 0.005, C = 15, verbose = 2)
    classifier2 = svc2.fit(xsample2, ysample2)
    
    ypred = classifier2.predict(x_te)  
    print(i, '\n', classification_report(y_te, ypred))
    arrr_f1.append(f1_score(y_te, ypred))

    cm = confusion_matrix(y_te, ypred)
    tp, tn = cm[1][1], cm[0][0]
    fp, fn = cm[0][1], cm[1][0]
    pod = tp/(tp+fn)
    far = fp/(tp+fp)
    pss = pod-(fp/(fp+tn))
    acc = (tp+tn)/(tp+tn+fp+fn)
    hr = (tp+fn)*(tp+fp)/(tp+fp+tn+fn)
    gss = (tp-hr)/(tp+fn+fp-hr)
    kp = (2*((tp*tn)-(fn*fp)))/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)) 
    exp = (1/len(x_te))*((tp+fn)*(tp+fp)+(tn+fn)*(tn+fp))
    hss = ((tp+tn)-exp)/(len(x_te)-exp)

    arrr_pod.append(pod)
    arrr_far.append(far)
    arrr_hss.append(hss)
    arrr_pss.append(pss)
    arrr_gss.append(gss)    


# In[61]:


xticks = np.arange(0, 122, 11)
metric_arr2 = [arrr_pod, arrr_far, arrr_f1, arrr_pss, arrr_gss, arrr_hss]
metric_labels2 = ['POD', 'FAR', 'f-1 Score', 'PSS', 'GSS', 'HSS']

fig, axes = plt.subplots(2, 3, figsize = (15, 8), sharex = True)
axes = axes.ravel()
warnings.filterwarnings("ignore")
print('\n')

for i in range(len(metric_arr2)):
    fig.suptitle('Performance metrics vs number of features based on Sequential Feature Selection', fontsize = 18)
    axes[i].plot(no_samples, metric_arr2[i], 'o-', linewidth = 1, markersize = 2.5)
    axes[i].set_ylabel(metric_labels[i], fontsize = 14, labelpad = 8)
    axes[i].tick_params(direction = 'in')
    #axes[i].set_xlim(0,122)
    
    if i >= 3:
        axes[i].set_xlabel('Number of features', fontsize = 14, labelpad = 8)
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels(xticks, fontsize = 12)
    else:
        axes[i].tick_params(direction = 'in')
        
    axes[i].set_yticklabels(['%.2f' % y for y in axes[i].get_yticks()], fontsize = 12)
    
plt.tight_layout()   
#plt.savefig('Atlantic_metrics_sfs.jpg', dpi = 1200)
plt.show()


# In[53]:


plt.figure(figsize = (10, 7))
#plt.title('POD and FAR vs number of features obtained from random forest', fontsize = 14)
plt.plot(no_samples, arrr_pod, 'o-', label = 'POD', color= 'blue', linewidth = 1, markersize = 3)
plt.plot(no_samples, arrr_far, 'o-', label = 'FAR', color = 'red', linewidth = 1, markersize = 3)
plt.xlabel('Number of features', fontsize = 14, labelpad = 8)
plt.yticks(fontsize = 12)
plt.xticks(xticks, fontsize = 12)
#plt.ylim(-0.05, 0.65)
plt.legend()
#plt.savefig('POD and FAR for atlantic sfs.jpg', dpi = 1200)
plt.show()


# TRAINING DATA PRE-PROCESSING

# In[335]:


cols_dvs = list(df1)    # Columns with target 
cols = list(df1)[:-1]   # Columns without target
df_train = df1[cols].astype(float)

scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)
xtrain = df_train_scaled[:]
ytrain = np.array(df1['dvs24'])


# In[336]:


cols_dvs4 = list(df4)    # Columns with target 
cols4 = list(df4)[:-1]   # Columns without target
df_train4 = df4[cols4].astype(float)

scaler = StandardScaler()
df_train_scaled4 = scaler.fit_transform(df_train4)
xtrain4 = df_train_scaled4[:]
ytrain4 = np.array(df4['dvs24'])


# In[337]:


for i in range(len(ytrain)):
    if ytrain[i] >= 30:
        ytrain[i] = 1
    else:
        ytrain[i] = 0
        
#ytrain[:500]


# In[338]:


# Total UNRI and RI cases in train data
zeroes = np.where(ytrain < 1)
ones = np.where(ytrain > 0)
a, b = ytrain[zeroes], ytrain[ones]
print(len(a))
print(len(b))


# SEQUENTIAL FEATURE SELECTION 

# In[10]:


smote = SMOTE(random_state = 42,
           sampling_strategy = {0: 9768, 1: 2442},    # Input the no. of UNRI and RI samples based on sampling ratio
           k_neighbors = 10)                        
xsampled, ysampled = smote.fit_resample(xtrain, ytrain)

svc_sfs = SVC(kernel = 'rbf', gamma = 0.001, C = 10, verbose = 2)
classifier_sfs = svc_sfs.fit(xsampled, ysampled)


# In[254]:


start = time.time()

sfs = SFS(classifier_sfs,
          k_features = 'best',
          forward = True,
          floating = False,
          scoring = 'f1',
          cv = 0,
          verbose = 2, 
          n_jobs = 6).fit(xsampled, ysampled)  # clone_estimator = False

end = time.time()
print('time elapsed: ', (end - start) / 60, 'min')


# In[293]:


features_obtained = sfs.k_feature_names_
print(sfs.k_score_)

index = features_obtained
#index = (120,)
#index = (0, 16, 28, 42, 56, 60, 62, 63, 64, 65, 74, 77, 84, 86, 88, 91, 93, 94, 97, 100, 104, 107, 112, 114, 120)

converted = []
for i in range(len(index)):
    converted.append(int(index[i]))

final_features = []
for j in range(len(converted)):
    final_features.append(feat[converted[j]])
final_features


# In[ ]:


metrics3 = sfs.get_metric_dict()
#print(metrics2)
df_metric3 = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
df_metric3


# In[ ]:


df_metric3.to_csv('sfs_atlantic_basin_f1.csv')


# In[23]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

fig1 = plot_sfs(sfs.get_metric_dict(confidence_interval = 0.95),
                kind = 'std_err', figsize = (10, 6))
plt.xticks([1, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121])
#plt.ylim(0.98, 1.00)
#plt.xlim(30, 122)
plt.show()


# In[ ]:





# In[129]:


#df3 = pd.DataFrame(df.iloc[44204:44662], columns = feat)   # WP 2017
#df3 = pd.DataFrame(df.iloc[54960:55370], columns = atlantic_22_rf)   # AL 2017
#df3 = pd.DataFrame(df.iloc[15547:15778], columns = indian_10_rf)   # IO 2015-2017
#df3 = pd.DataFrame(df.iloc[13062:13397], columns = east_36_rf)   # EP 2017

#df3 = pd.DataFrame(pd.read_csv("NOAA_operational_vars_global_w_dvs24.csv"))  # operational AL 2010-16
#df3 = pd.DataFrame(df3, columns = feat)
#df3 = df3.dropna(axis = 1)

#df3 = pd.DataFrame(pd.read_csv("atlantic_2019-20.csv"))  # operational AL 2019-20
#df3 = pd.DataFrame(df3, columns = feat)
df3


# In[339]:


cols2 = list(df3)[:-1]
df_test = df3[cols2].astype(float)

df_test_scaled = scaler.fit_transform(df_test)
xtest = df_test_scaled[:]
ytest = np.array(df3['dvs24'])


# In[340]:


for i in range(len(ytest)):
    if ytest[i] >= 30:
        ytest[i] = 1
    else:
        ytest[i] = 0
        
#print(ytest[:1000])


# In[341]:


# Total UNRI and RI cases in test data
zeroes = np.where(ytrain < 1)
ones = np.where(ytrain > 0)
a, b = ytrain[zeroes], ytrain[ones]
print(len(a))
print(len(b))


# SMOTE FOR HANDLING CLASS IMBALANCE

# In[342]:


#nm = NearMiss()
#xr, yr = nm.fit_resample(xtrain, ytrain)

sm = SMOTE(random_state = 42,
           sampling_strategy = {0: 2020, 1: 2020},    # Input the no. of UNRI and RI samples based on sampling ratio
           k_neighbors = 10)                        
xs, ys = sm.fit_resample(xtrain, ytrain)


# In[343]:


print(xs.shape)
print(ys.shape)


# SVM TRAINING AND TESTING

# In[344]:


start = time.time()

svc = SVC(kernel = 'rbf', gamma = 0.005, C = 15, verbose = 2)
classifier = svc.fit(xs, ys)

end = time.time()
print('Time elapsed:', (end - start) / 60, 'min')


# In[45]:


ypred_svc = classifier.decision_function(xtest)


# In[345]:


ypredict = classifier.predict(xtest)  
#print(ypredict)
print(classification_report(ytest, ypredict))


# In[346]:


cm = confusion_matrix(ytest, ypredict)
display = ConfusionMatrixDisplay(cm, display_labels = ['UNRI', 'RI'])
display.plot()
#plt.savefig('conf_matrix_IO_ppt1.jpg', dpi = 1200)
plt.show()


# In[347]:


tp, tn = cm[1][1], cm[0][0]
fp, fn = cm[0][1], cm[1][0]
pod = tp/(tp+fn)
far = fp/(tp+fp)
pss = pod-(fp/(fp+tn))
acc = (tp+tn)/(tp+tn+fp+fn)
hr = (tp+fn)*(tp+fp)/(tp+fp+tn+fn)
gss = (tp-hr)/(tp+fn+fp-hr)
kp = (2*((tp*tn)-(fn*fp)))/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)) 
exp = (1/len(xtest))*((tp+fn)*(tp+fp)+(tn+fn)*(tn+fp))
hss = ((tp+tn)-exp)/(len(xtest)-exp)

print('POD:',pod)
print('FAR:',far)
print('Kappa:',kp)
print('PSS:',pss)
print('GSS:',gss)
print('HSS:',hss)   
print('Accuracy:',acc)


# In[77]:


svc_fpr, svc_tpr, threshold = roc_curve(ytest, ypred_svc)
svc_fpr2, svc_tpr2, threshold2 = roc_curve(ytest5, ypred_svc2)
auc_svc = auc(svc_fpr, svc_tpr)
auc_svc2 = auc(svc_fpr2, svc_tpr2)

plt.figure(figsize=(6, 6))
plt.plot(svc_fpr, svc_tpr, label='Atlantic (AUC = %0.3f)' % auc_svc)
plt.plot(svc_fpr2, svc_tpr2, label='Indian (AUC = %0.3f)' % auc_svc2)
plt.xlabel('False Positive Rate', fontsize = 14, labelpad = 8)
plt.ylabel('True Positive Rate', fontsize = 14, labelpad = 8)
plt.tick_params(direction = 'in')
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)0
plt.legend(loc = 'lower right', bbox_to_anchor = (1.00, 0.05))
#plt.savefig('ROC-AUC.jpg', dpi = 1200)
plt.show()


# In[ ]:





# In[348]:


from explainerdashboard import ClassifierExplainer, ExplainerDashboard


# In[ ]:


import types
start = time.time()

cols_dvs = list(df1)    # Columns with target 
cols = list(df1)[:-1]   # Columns without target
df_train = df1[cols].astype(float)

scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)
xtrain = df_train_scaled[:]
ytrain = df1['dvs24'].apply(lambda x: 1 if float(x) >= 30 else 0)

cols2 = list(df3)[:-1]
df_test = df3[cols2].astype(float)

df_test_scaled = scaler.fit_transform(df_test)
xtest = df_test_scaled[:]
ytest = df3['dvs24'].apply(lambda x: 1 if float(x) >= 30 else 0)

sm = SMOTE(random_state = 42, sampling_strategy = {0: 2020, 1: 2020}, k_neighbors = 10)                        
xs, ys = sm.fit_resample(xtrain, ytrain)

svc = SVC(kernel = 'rbf', gamma = 0.005, C = 15, verbose=2)
classifier = svc.fit(xs, ys)

def predict_proba(self, X):
    pred = self.predict(X)
    return np.array([1-pred, pred]).T
svc.predict_proba = types.MethodType(predict_proba, svc)

classifier = svc.fit(xs, ys)

xtest_df = pd.DataFrame(xtest, columns=df3.columns[:-1])
explainer = ClassifierExplainer(classifier, xtest_df, ytest)

db = ExplainerDashboard(explainer, model_summary=True, shap_interaction=True, shap_dependance=True)
db.run()

end = time.time()
print('time elapsed:', (end-start)/60)


# In[285]:


get_ipython().system('pip install shap')


# In[286]:


import shap


# In[287]:


# Compute SHAP values for the training data
explainer = shap.Explainer(classifier.predict, xs)
shap_values_train = explainer(xs)

# Compute SHAP values for the test data
shap_values_test = explainer(df_test_scaled)

# Get the SHAP feature importances
shap.summary_plot(shap_values_train, xtrain, plot_type="bar")


# In[ ]:





# In[50]:


start = time.time()

# train SVC model
svc = SVC(kernel='rbf', gamma=0.002, C=15, verbose=2)
classifier = svc.fit(xs, ys)

# apply predict_proba monkey patch
def predict_proba(self, X):
    pred = self.predict(X)
    return np.array([1-pred, pred]).T

svc.predict_proba = types.MethodType(predict_proba, svc)

# create explainer
xtest_df = pd.DataFrame(xtest)  # convert numpy ndarray to pandas DataFrame
explainer = ClassifierExplainer(svc, xtest_df, ytest)

# create dashboard
dashboard_title = "My SVC Dashboard"
db = ExplainerDashboard(explainer, title=dashboard_title)
db.run()

end = time.time()
print('time elapsed:', (end-start)/60)


# In[ ]:


explainer2 = shap.KernelExplainer(classifier.predict_proba, xs)
shap_values = explainer2.shap_values(xs)

feature_name = 'SHGC_t24'
feature_index = cols.index(feature_name)

shap.dependence_plot(feature_index, shap_values, xs, show = False)
plt.xlabel(feature_name)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[937]:


start = time.time()

def svc(xs, ys):
    svc = SVC(kernel = 'rbf', gamma = 0.001, C = 5, verbose = 2)
    classifier = svc.fit(xs, ys)
    return classifier

pod_sample, far_sample, f1_sample, hss_sample = [], [], [], []

for split in range(20):
    print('\n', 'TRAIN SPLIT', split + 1, '\n', '===========================================================')    
    x_train_copy, y_train_copy = xs.copy(), ys.copy()
    x_test_copy, y_test_copy = xtest.copy(), ytest.copy()

    idx = np.random.choice(len(x_train_copy), size = int(len(x_train_copy) * 0.1), replace = False)

    x_train_sub = np.delete(x_train_copy, idx, axis = 0)
    y_train_sub = np.delete(y_train_copy, idx, axis = 0) 
    y_original = np.array(df3['dvs24'])

    model = svc(x_train_sub, y_train_sub)
    yhat = model.predict(x_test_copy)
    cm = confusion_matrix(y_test_copy, yhat)
    tp, tn = cm[1][1], cm[0][0]
    fp, fn = cm[0][1], cm[1][0]
    pod = tp / (tp + fn)
    far = fp / (tp + fp)
    f1 = f1_score(y_test_copy, yhat)
    exp = (1/len(x_test_copy))*((tp+fn)*(tp+fp)+(tn+fn)*(tn+fp))
    hss = ((tp+tn)-exp)/(len(x_test_copy)-exp)

    pod_sample.append(pod)
    far_sample.append(far)
    f1_sample.append(f1)
    hss_sample.append(hss)
    print('>%d, POD: %.3f' % (split + 1, pod), 'FAR:', far, 'f-1:', f1, 'HSS', hss)

pod_array, far_array = np.asarray(pod_sample), np.asarray(far_sample) 
f1_array, hss_array = np.asarray(f1_sample), np.asarray(hss_sample)

pod_mean, far_mean, f1_mean, hss_mean = pod_array.mean(), far_array.mean(), f1_array.mean(), hss_array.mean()
pod_std, far_std, f1_std, hss_std = pod_array.std(), far_array.std(), f1_array.std(), hss_array.std()
pod_interval, far_interval, f1_interval, hss_interval = 0.468 * pod_std, 0.468 * far_std, 0.468 * f1_std, 0.468 * hss_std
pod_lower, pod_upper = pod_mean - pod_interval, pod_mean + pod_interval
far_lower, far_upper = far_mean - far_interval, far_mean + far_interval
f1_lower, f1_upper = f1_mean - f1_interval, f1_mean + f1_interval
hss_lower, hss_upper = hss_mean - hss_interval, hss_mean + hss_interval

print('\n')
print('POD mean:', pod_mean, 'FAR mean:', far_mean, 'f-1 mean:', f1_mean, 'HSS mean:', hss_mean)
#print('POD std:', pod_std, 'FAR std:', far_std, 'f-1 std:', f1_std)
print('95%% POD confidence interval: [%.3f, %.3f]' % (pod_lower, pod_upper))
print('95%% FAR confidence interval: [%.3f, %.3f]' % (far_lower, far_upper))
print('95%% f-1 confidence interval: [%.3f, %.3f]' % (f1_lower, f1_upper))
print('95%% HSS confidence interval: [%.3f, %.3f]' % (hss_lower, hss_upper))

end = time.time()
print('\n', "Time elapsed:", (end - start)/60, 'min')    


# In[155]:


fig, ax = plt.subplots(2, 2, sharex = True)

number = np.arange(1, 21, 1)
ax[0, 0].plot(number, pod_sample)

ax[0, 1].plot(number, far_sample)

ax[1, 0].plot(number, f1_sample)

ax[1, 1].plot(number, hss_sample)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


count = 0
    s0 ,s1 = xtest.shape[0], xtest.shape[1]
    while count < 0.1 * (s0 * s1):
        k = np.random.randint(s0)
        l = np.random.randint(s1)
        xtest[k][l] = 0        # SVC DOES NOT ACCEPT NaN INPUT, HENCE 'ZERO' IS THE CURRENT INPUT!
        count += 1       


# In[67]:


test_splits, ytest_arr_splits = [], []
for i in range(0, 410, 41):
    test_data = df3.to_numpy()
    ytest_arr = test_data[:, -1]
    ytest_arr_splits.append(np.delete(ytest_arr, slice(i, i + 41)))
    test_data_scaled = scaler.fit_transform(test_data)
    test_splits.append(np.delete(test_data_scaled, slice(i, i + 41), axis = 0))
    
x_test, y_test = test_splits[2][:, :-1], ytest_arr_splits[2]
print(x_test.shape)
print(y_test.shape)
print(y_test)


# In[135]:


arr = np.random.randint(1, 9, size = (10, 5))
print(arr)
arr_splits = []
for i in range(0, len(arr), 2):
    print(i, i+2)
    arr_splits.append(np.delete(arr, slice(i, i+2), axis = 0))


# In[183]:


xtest_splits, ytest_splits = [], []

for i in range(0, df3.shape[0], 41):
    test_data = df3.to_numpy()
    ytest_arr = test_data[:, -1]
    xtest_data_scaled = scaler.fit_transform(test_data[:, :-1])
    xtest_splits.append(np.delete(xtest_data_scaled, slice(i, i + 41), axis = 0))
    ytest_splits.append(np.delete(ytest_arr, slice(i, i + 41)))

for split in range(10):
    print('\n', 'TEST SPLIT', split + 1, '\n', '=====================================')
    x_test, y_test = test_splits[split][::], ytest_splits[split]   
    
    for i in range(len(y_test)):
        if y_test[i] >= 30:
            y_test[i] = 1
        else:
            y_test[i] = 0


# In[129]:


start = time.time()

def svc(xs, ys):
    svc = SVC(kernel = 'rbf', gamma = 0.001, C = 5, verbose = 2)
    classifier = svc.fit(xs, ys)
    return classifier

percent, xtest_splits, ytest_splits = int(len(df3)/10),  [], []

for i in range(0, len(df3), percent):
    test_data = df3.to_numpy()
    ytest_arr = test_data[:, -1]
    xtest_data_scaled = scaler.fit_transform(test_data[:, :-1])
    xtest_splits.append(np.delete(xtest_data_scaled, slice(i, i + percent), axis = 0))
    ytest_splits.append(np.delete(ytest_arr, slice(i, i + percent)))     

pod_sample, far_sample, f1_sample, hss_sample = [], [], [], []

for split in range(10):
    print('\n', 'TEST SPLIT', split + 1, '\n', '=====================================')
    x_test, y_test = xtest_splits[split][::], ytest_splits[split]   
    
    for i in range(len(y_test)):
        if y_test[i] >= 30:
            y_test[i] = 1
        else:
            y_test[i] = 0

    model = svc(xs, ys)
    yhat = model.predict(x_test)
    cm = confusion_matrix(y_test, yhat)
    tp, tn = cm[1][1], cm[0][0]
    fp, fn = cm[0][1], cm[1][0]
    pod = tp / (tp + fn)
    far = fp / (tp + fp)
    f1 = f1_score(y_test, yhat)
    exp = (1/len(x_test))*((tp+fn)*(tp+fp)+(tn+fn)*(tn+fp))
    hss = ((tp+tn)-exp)/(len(x_test)-exp)

    pod_sample.append(pod)
    far_sample.append(far)
    f1_sample.append(f1)
    hss_sample.append(hss)
    print('>%d, POD: %.3f' % (split + 1, pod), 'FAR:', far, 'f-1:', f1, 'HSS', hss)

pod_array, far_array = np.asarray(pod_sample), np.asarray(far_sample) 
f1_array, hss_array = np.asarray(f1_sample), np.asarray(hss_sample)

pod_mean, far_mean, f1_mean, hss_mean = pod_array.mean(), far_array.mean(), f1_array.mean(), hss_array.mean()
pod_std, far_std, f1_std, hss_std = pod_array.std(), far_array.std(), f1_array.std(), hss_array.std()
pod_interval, far_interval, f1_interval, hss_interval = 0.715 * pod_std, 0.715 * far_std, 0.715 * f1_std, 0.715 * hss_std
pod_lower, pod_upper = pod_mean - pod_interval, pod_mean + pod_interval
far_lower, far_upper = far_mean - far_interval, far_mean + far_interval
f1_lower, f1_upper = f1_mean - f1_interval, f1_mean + f1_interval
hss_lower, hss_upper = hss_mean - hss_interval, hss_mean + hss_interval

print('\n')
print('POD mean:', pod_mean, 'FAR mean:', far_mean, 'f-1 mean:', f1_mean, 'HSS mean:', hss_mean)
#print('POD std:', pod_std, 'FAR std:', far_std, 'f-1 std:', f1_std)
print('95%% POD confidence interval: [%.3f, %.3f]' % (pod_lower, pod_upper))
print('95%% FAR confidence interval: [%.3f, %.3f]' % (far_lower, far_upper))
print('95%% f-1 confidence interval: [%.3f, %.3f]' % (f1_lower, f1_upper))
print('95%% HSS confidence interval: [%.3f, %.3f]' % (hss_lower, hss_upper))

end = time.time() 
print('Time elapsed:', (end - start) / 60, 'min')


# In[64]:


basin = [1, 2, 3, 4]
avg_pod = np.array([0.520, 0.466, 0.691, 0.534])
avg_far = np.array([0.280, 0.590, 0.452, 0.502])
avg_f1 = np.array([0.602, 0.435, 0.601, 0.514])
avg_hss = np.array([0.563, 0.387, 0.581, 0.468])
lb_pod = np.array([0.498, 0.433, 0.641, 0.509])
ub_pod = np.array([0.544, 0.490, 0.741, 0.559])
lb_far = np.array([0.268, 0.577, 0.429, 0.482])
ub_far = np.array([0.293, 0.605, 0.476, 0.522])
lb_f1 = np.array([0.587, 0.418, 0.584, 0.495])
ub_f1 = np.array([0.619, 0.453, 0.618, 0.534])
lb_hss = np.array([0.548, 0.369, 0.566, 0.449])
ub_hss = np.array([0.579, 0.405, 0.597, 0.489])

pod_err = []
far_err = []
#f1_err  = [(), (), (), ()]
#hss_err = [(), (), (), ()]


# In[65]:


combined_array = np.column_stack((basin, avg_pod, avg_far, avg_f1, avg_hss, lb_pod, ub_pod, lb_far, ub_far, lb_f1, ub_f1, lb_hss, ub_hss))
df_combined = pd.DataFrame(combined_array, columns = ['Basins','Mean POD','Mean FAR','Mean F1','Mean HSS','LB POD','UB POD','LB FAR','UB FAR','LB F1','UB F1','LB HSS','UB HSS'])
print("combined dataframe:\n", df_combined)


# In[66]:


fig, axs = plt.subplots(2, 2, figsize = (9, 7), sharex = True)

x = df_combined['Basins']

axs[0, 0].plot(x, df_combined['Mean POD'], 'o-', color = 'g')
axs[0, 0].fill_between(x, df_combined['LB POD'], df_combined['UB POD'], color = 'g', alpha = .15)
#axs[0, 0].errorbar(x, df_combined['Mean POD'], yerr = pod_err, markerfacecolor = 'black', markeredgecolor = 'blue')
#axs[0, 0].set_xlabel('Basin', fontsize = 12)
#ax[].set_xticks()
axs[0, 0].set_ylabel('POD', fontsize = 14)

axs[0, 1].plot(x, df_combined['Mean FAR'], 'o-', color = 'r')
axs[0, 1].fill_between(x, df_combined['LB FAR'], df_combined['UB FAR'], color = 'r', alpha = .15)
#axs[0, 1].set_xlabel('Basin', fontsize = 12)
axs[0, 1].set_ylabel('FAR', fontsize = 14)

axs[1, 0].plot(x, df_combined['Mean F1'], 'o-', color = 'm')
axs[1, 0].fill_between(x, df_combined['LB F1'], df_combined['UB F1'], color = 'm', alpha = .15)
axs[1, 0].set_xlabel('Basin', fontsize = 14)
axs[1, 0].set_ylabel('f-1 Score', fontsize = 14)
axs[1, 0].set_xticklabels(['Atlantic', 'West Pacific', 'Indian', 'East Pacific'])

axs[1, 1].plot(x, df_combined['Mean HSS'], 'o-', color = 'b')
axs[1, 1].fill_between(x, df_combined['LB HSS'], df_combined['UB HSS'], color = 'b', alpha = .15)
axs[1, 1].set_xlabel('Basin', fontsize = 14)
axs[1, 1].set_ylabel('HSS', fontsize = 14)
axs[1, 1].set_xticklabels(['Atlantic', 'West Pacific', 'Indian', 'East Pacific'])

#for ax in axs.flat:
#ax.set(xlabel = 'Basin')
#ax.label_outer()

plt.tight_layout()
#for ax in axs.flat:
#    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.



#fig, ax = plt.subplots(figsize = (10, 6))
#x = df_combined['Index']
#ax.plot(x, df_combined['Mean POD'], 'o-')
#ax.fill_between(x, df_combined['LB POD'], df_combined['UB POD'], color = 'b', alpha = .15)
#plt.xlabel('Models tested', fontsize = 14)
#plt.ylabel('POD', fontsize = 14)
#plt.xticks(np.arange(1, 11, 1))
#plt.yticks(np.arange(0.60, 0.80, 0.02))
#plt.ylim(0.59, 0.81)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[787]:


st = time.time()

x = []
#ratio = [len(b)/len(a), 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 1.00]
ratio = [0.25, 0.30, 0.35, 0.40, 0.50]

for i in range(len(ratio)):
    x.append(round(len(a) * ratio[i]))
    
gm = [0.001, 0.002, 0.004, 0.005, 0.006, 0.008, 0.01]
C = [5, 10, 15, 20]

pod_arr, far_arr, pss_arr, gss_arr, hss_arr = [], [], [], [], []
f1 = []

count = 0
for i in range(len(x)):
    print('Ratio:', x[i])
    for j in range(len(gm)):
        print('Gamma:', gm[j])
        for k in range(len(C)):
            print('C value:', C[k])
            sm = SMOTE(random_state = 42, sampling_strategy = {0: len(a), 1: x[i]}, k_neighbors = 10)
            xs, ys = sm.fit_resample(xtrain, ytrain)
            svc = SVC(kernel = 'rbf', gamma = gm[j], C = C[k])
            classifier = svc.fit(xs, ys)
            ypredict = classifier.predict(xtest)  
            print(count, '\n', classification_report(ytest, ypredict))
            f1.append(f1_score(ytest, ypredict))
            
            cm = confusion_matrix(ytest, ypredict)
            tp, tn = cm[1][1], cm[0][0]
            fp, fn = cm[0][1], cm[1][0]
            pod = tp/(tp+fn)
            far = fp/(tp+fp)
            pss = pod-(fp/(fp+tn))
            acc = (tp+tn)/(tp+tn+fp+fn)
            hr = (tp+fn)*(tp+fp)/(tp+fp+tn+fn)
            gss = (tp-hr)/(tp+fn+fp-hr)
            kp = (2*((tp*tn)-(fn*fp)))/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)) 
            exp = (1/len(xtest))*((tp+fn)*(tp+fp)+(tn+fn)*(tn+fp))
            hss = ((tp+tn)-exp)/(len(xtest)-exp)
            
            pod_arr.append(pod)
            far_arr.append(far)
            hss_arr.append(hss)
            pss_arr.append(pss)
            gss_arr.append(gss)

            count += 1
        print("===========================================================")
        
en = time.time()
print('Time elapsed:', (en - st) / 60, 'min')


# In[793]:


no_samples = np.arange(1, 141, 1)

plt.plot(no_samples, f1, 'o-', markersize = 3)
plt.vlines(x = 57, ymin = 0, ymax = 0.6, color = 'black')
plt.xlabel('Number of models', fontsize = 10)
plt.ylabel('f1 score', fontsize = 10)
plt.show()#


# In[338]:


#plt.figure(figsize = (6, 5))
no_samples = np.arange(1, 316, 1)

plt.plot(no_samples, pod_arr)
plt.xlabel('Number of models', fontsize = 10)
plt.ylabel('POD', fontsize = 10)
plt.show()

plt.plot(no_samples, far_arr)
plt.xlabel('Number of models', fontsize = 10)
plt.ylabel('FAR', fontsize = 10)
plt.show()

plt.plot(no_samples, pss_arr)
plt.xlabel('Number of models', fontsize = 10)
plt.ylabel('PSS', fontsize = 10)
plt.show()

plt.plot(no_samples, gss_arr)
plt.xlabel('Number of models', fontsize = 10)
plt.ylabel('GSS', fontsize = 10)
plt.show()

plt.plot(no_samples, hss_arr)
plt.xlabel('Number of models', fontsize = 10)
plt.ylabel('HSS', fontsize = 10)
plt.show()


# In[199]:


st = time.time()

x, f1 = [], []
ratio = np.arange(0.10, 1.05, 0.05)
for i in range(len(ratio)):
    x.append(round(len(a) * ratio[i]))
    
for i in range(len(x)):
    sm = SMOTE(random_state = 42, sampling_strategy = {0: len(a), 1: x[i]}, k_neighbors = 10)
    xs, ys = sm.fit_resample(xtrain, ytrain)
    svc = SVC(kernel = 'rbf', gamma = 0.001, C = 10)
    classifier = svc.fit(xs, ys)
    ypredict = classifier.predict(xtest)  
    print(classification_report(ytest, ypredict))
    f1.append(f1_score(ytest, ypredict))
print("===========================================================")
        
en = time.time()
print('Time elapsed:', (en - st) / 60, 'min')


# In[209]:


st = time.time()

x2, f1_2 = [], []
ratio2 = np.arange(0.10, 1.05, 0.05)
for i in range(len(ratio2)):
    x2.append(round(len(a) * ratio2[i]))
    
for i in range(len(x2)):
    sm2 = SMOTE(random_state = 42, sampling_strategy = {0: len(a), 1: x2[i]}, k_neighbors = 10)
    xs, ys = sm2.fit_resample(xtrain, ytrain)
    svc = SVC(kernel = 'rbf', gamma = 0.001, C = 5)
    classifier = svc.fit(xs, ys)
    ypredict = classifier.predict(xtest)  
    print(classification_report(ytest, ypredict))
    f1_2.append(f1_score(ytest, ypredict))
print("===========================================================")
        
en = time.time()
print('Time elapsed:', (en - st) / 60, 'min')


# In[220]:


st = time.time()

x3, f1_3 = [], []
ratio3 = np.arange(0.10, 1.05, 0.05)
for i in range(len(ratio3)):
    x3.append(round(len(a) * ratio3[i]))
    
for i in range(len(x3)):
    print('RATIO:', ratio3[i])
    sm3 = SMOTE(random_state = 42, sampling_strategy = {0: len(a), 1: x3[i]}, k_neighbors = 10)
    xs, ys = sm3.fit_resample(xtrain, ytrain)
    svc = SVC(kernel = 'rbf', gamma = 0.005, C = 15)
    classifier = svc.fit(xs, ys)
    ypredict = classifier.predict(xtest)  
    print(classification_report(ytest, ypredict))
    f1_3.append(f1_score(ytest, ypredict))
print("===========================================================")
        
en = time.time()
print('Time elapsed:', (en - st) / 60, 'min')


# In[92]:


st = time.time()

x4, f1_4 = [], []
ratio4 = np.arange(0.1, 1.1, 0.1)
for i in range(len(ratio4)):
    x4.append(round(len(a) * ratio4[i]))
    
for i in range(len(x4)):
    print('RATIO:', ratio4[i])
    sm4 = SMOTE(random_state = 42, sampling_strategy = {0: len(a), 1: x4[i]}, k_neighbors = 10)
    xs, ys = sm4.fit_resample(xtrain, ytrain)
    svc = SVC(kernel = 'rbf', gamma = 0.001, C = 10)
    classifier = svc.fit(xs, ys)
    ypredict = classifier.predict(xtest)  
    print(classification_report(ytest, ypredict))
    f1_4.append(f1_score(ytest, ypredict))
print("===========================================================")
        
en = time.time()
print('Time elapsed:', (en - st) / 60, 'min')


# In[222]:


plt.figure(figsize = (10, 6))
plt.plot(ratio, f1, 'o-', label = 'Atlantic')
plt.plot(ratio2, f1_2, 's-', label = 'Western Pacific')
plt.plot(ratio3, f1_3, 'd-', label = 'Indian')
#plt.plot(ratio4, f1_4, 'o-', label = 'Atlantic 2')
plt.xlabel('Class ratio', fontsize = 15, labelpad = 8)
plt.ylabel('f-1 score', fontsize = 15, labelpad = 8)
plt.xticks(np.arange(0.1, 1.1, 0.1), fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylim(-0.05, 0.85)
plt.legend()
#plt.savefig('RI_classratio_plot2.jpg', dpi = 1200)
plt.show()


# In[ ]:





# In[ ]:





# In[974]:


X = ['Yang', 'Su et al.', 'Wei and Yang', 'Present work']
x = np.arange(len(X))
width = 0.25

scores = np.array([[0.34, 0.711], [0.57, 0.90], [0.411, 0.621], [0.65, 0.35]])

plt.bar(x - width/2, scores[:, 0], width=width, label='POD')
plt.bar(x + width/2, scores[:, 1], width=width, label='FAR')
plt.ylim(0.0, 1.00)

plt.xticks(x, X)
plt.legend()
plt.show()


# In[989]:


X = ['Yang.', 'Su et al.', 'Wei and Yang', 'Present work']
x = np.arange(len(X))
y = np.arange(0.0, 1.2, 0.2)
width = 0.25

scores = np.array([[0.34, 0.711], [0.57, 0.90], [0.411, 0.621], [0.65, 0.35]])

fig, ax = plt.subplots(figsize = (8, 6))
ax.bar(x - width/2, scores[:, 0], width = width, label = 'POD')
ax.bar(x + width/2, scores[:, 1], width = width, label = 'FAR', color = 'red')
#ax.set_ylim(0.0, 1.00)

ax.set_xticks(x)
ax.set_yticks(y)
ax.set_xticklabels(X, fontsize = 14)
ax.set_yticklabels(y, fontsize = 14)
ax.tick_params(axis = 'both', which = 'major', direction = 'in')

plt.legend()
plt.show()


# In[4]:


X = ['Yang', 'Su et al.', 'Wei and Yang', 'Present work']
x = np.arange(len(X))
width = 0.25

scores = np.array([[0.34, 0.711], [0.57, 0.90], [0.411, 0.621], [0.65, 0.35]])

fig, ax = plt.subplots(figsize = (8, 6))
ax.bar(x - width/2, scores[:, 0], width = width, label = 'POD', color = 'blue')
ax.bar(x + width/2, scores[:, 1], width = width, label = 'FAR', color = 'red')

ax.set_xticks(x)
ax.set_xticklabels(X, fontsize = 14)

# set y-ticks and their labels
y_ticks = np.arange(0.0, 1.1, 0.1)
y_ticklabels = [round(y_tick, 2) for y_tick in y_ticks]
#minor_y_ticks = np.arange(0.1, 1.1, 0.1)
#ax.set_yticks(minor_y_ticks, minor=True)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticklabels, fontsize = 14)

ax.tick_params(axis = 'both', which = 'both', direction = 'in', pad = 9)
#ax.tick_params(axis = 'both', which = 'major', length = 5)
#ax.tick_params(axis = 'y', which = 'minor', length = 2.5)

plt.legend()
#plt.savefig('comparison2.jpg', dpi = 1200)
plt.show()


# In[8]:


X = ['Yang (2016)', 'Su et al.', 'Wei and Yang', 'Present work']
x = np.arange(len(X))
width = 0.25

scores = np.array([[0.34, 0.711], [0.57, 0.90], [0.411, 0.621], [0.65, 0.35]])

fig, ax = plt.subplots(figsize = (8, 6))
ax.bar(x - width/2, scores[:, 0], width = width, label = 'POD', color = 'blue')
ax.bar(x + width/2, scores[:, 1], width = width, label = 'FAR', color = 'red')

ax.set_xticks(x)
ax.set_xticklabels(X, fontsize = 14)

y_ticks = np.arange(0.0, 1.1, 0.1)
y_ticklabels = [round(y_tick, 2) for y_tick in y_ticks]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticklabels, fontsize = 14)

ax.tick_params(axis = 'both', which = 'both', direction = 'in', pad = 9)

# add text labels on top of the bars
for i in range(len(X)):
    ax.text(x[i] - width/2, scores[i, 0] + 0.01, str(round(scores[i, 0], 2)), ha = 'center', fontsize = 12)
    ax.text(x[i] + width/2, scores[i, 1] + 0.01, str(round(scores[i, 1], 2)), ha = 'center', fontsize = 12)

plt.legend()
plt.savefig('comparison2.jpg', dpi = 1200)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

skfold = StratifiedKFold(n_splits = 5)
model = SVC(kernel = 'rbf', gamma = 0.001, C = 10, verbose = True)
scores = cross_val_score(model, xtrain, ytrain, cv = skfold)
model.fit(xtrain, ytrain)
print('\n',scores)
print('\n',np.mean(scores))


# In[51]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(model, xtest, ytest )


# In[156]:


start = time.time()

svc = SVC(kernel = 'rbf', gamma = 0.001, C = 10, verbose = 2)
classifier = svc.fit(xs, ys)

end = time.time()
print('Time elapsed:', (end - start) / 60, 'min')


# In[157]:


ypredict = classifier.predict(xtest)  
#print(ypredict)
print(classification_report(ytest, ypredict))


# In[16]:


#print(ytest)
#print(ypredict)
plt.hist(ytest, label = 'Original')
#plt.show()
plt.hist(ypredict, label = 'Prediction')
plt.xticks([0, 1])
plt.xlim(-0.25,1.25)
plt.ylim(0, 425)
plt.xlabel('Class', fontsize = 12)
plt.ylabel('No. of samples', fontsize = 12)
plt.legend()
plt.show()


# In[158]:


cm = confusion_matrix(ytest, ypredict)
display = ConfusionMatrixDisplay(cm, display_labels = ['UNRI', 'RI'])
display.plot()
#plt.savefig('conf_matrix_IO_ppt1.jpg', dpi = 1200)
plt.show()


# In[159]:


# Metrics... 

tp, tn = cm[1][1], cm[0][0]
fp, fn = cm[0][1], cm[1][0]
pod = tp/(tp+fn)
far = fp/(tp+fp)
pss = pod-(fp/(fp+tn))
acc = (tp+tn)/(tp+tn+fp+fn)
hr = (tp+fn)*(tp+fp)/(tp+fp+tn+fn)
gss = (tp-hr)/(tp+fn+fp-hr)
kp = (2*((tp*tn)-(fn*fp)))/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)) 
exp = (1/len(xtest))*((tp+fn)*(tp+fp)+(tn+fn)*(tn+fp))
hss = ((tp+tn)-exp)/(len(xtest)-exp)

print('POD:',pod)
print('FAR:',far)
print('Kappa:',kp)
print('PSS:',pss)
print('GSS:',gss)
print('HSS:',hss)   
print('Accuracy:',acc)


# # Sequential Feature Selection

# In[43]:


svc = SVC(kernel = 'rbf', gamma = 0.005, C = 15, verbose = False)

st = time.time()

sfs = SFS(estimator = svc,
          k_features = 'best',
          forward = True,
          floating = False,
          verbose = 2,
          scoring = 'average_precision',
          cv = 0,
          n_jobs = -1)

sfs1 = sfs.fit(xs, ys)

en = time.time()
print('Time elapsed:', (en - st)/60, 'min')


# In[44]:


print(sfs1.k_feature_names_)
print('\n','BEST SCORE:',sfs1.k_score_)


# In[51]:


index = sfs.k_feature_names_
#index = featrues
converted = []
for i in range(len(index)):
    converted.append(int(index[i]))

final_features = []
for j in range(len(converted)):
        final_features.append(feat[converted[j]])
final_features


# In[46]:


sfs1.get_metric_dict()


# In[47]:


df_feat = pd.DataFrame.from_dict(sfs1.get_metric_dict()).T
#df_feat[['Feature index', 'Average score']]
df_feat


# In[48]:


df_feat.to_csv('indianocean_svc_avgprecision_features_forward.csv')


# In[58]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

fig1 = plot_sfs(sfs1.get_metric_dict(confidence_interval = 0.95),
                kind = 'std_err', figsize = (10, 6), color= 'red')
#xticks = [1, 15, 30, 45, 60, 75, 90, 105, 120]
xticks = [1, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121]
#yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#yticks = np.arange(0.70, 1.00, 0.05)
plt.xticks(xticks, fontsize = 12)
plt.yticks(np.arange(0.80, 1.05, 0.05), fontsize = 12)
plt.xlabel('Number of features', fontsize = 14, labelpad = 8)
plt.ylabel(r'$ \mathrm{Performance\ (avg.\ precision)}$', fontsize = 14, labelpad = 8)
#plt.ylim(-0.075, 0.875)  # For Atlantic
plt.ylim(0.775, 1.025)   # For Indian
#plt.xlim(30, 122)
#plt.savefig('svc_indian_avgprecision_forward.jpg', dpi = 1200)
plt.show()


# In[20]:


no = np.arange(1, 31, 1)

#score = np.array([-0.619, -0.562, -0.498, -0.468, -0.449,
#                  -0.433, -0.421, -0.414, -0.408, -0.404, 
#                  -0.401, -0.399, -0.396, -0.394, -0.391])

score = np.array([-0.608, -0.557, -0.492, -0.455, -0.425, 
                  -0.406, -0.394, -0.388, -0.381, -0.376,
                  -0.373, -0.370, -0.365, -0.363, -0.359,
                  -0.358, -0.356, -0.355, -0.350, -0.350,
                  -0.346, -0.344, -0.342, -0.340, -0.338,
                  -0.337, -0.336, -0.331, -0.331, -0.331])

plt.plot(no, score)
plt.xlabel('Number of features', fontsize = 12)
plt.ylabel('Negative MAE', fontsize = 12)
plt.ylim(-0.63, 0)
#plt.savefig('no_feat.jpg', dpi = 1200)
plt.show()


# In[ ]:





# In[ ]:





# # GridsearchCV

# In[ ]:


st = time.time()
param_grid = [{'C': [10,20,30],
               'kernel': ['rbf'], 'gamma': [0.01]}]
classifier = SVC()

grid = GridSearchCV(estimator = classifier,
                    param_grid = param_grid,
                    scoring = 'f1',
                    cv = 5,
                    n_jobs = 6,
                    verbose = True)

grid.fit(xtrain, ytrain)
en = time.time()
print('time:', (en - st) / 60, 'min')


# In[31]:


grid.best_score_


# In[32]:


grid.best_params_


# In[ ]:




