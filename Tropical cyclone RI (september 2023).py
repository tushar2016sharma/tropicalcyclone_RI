#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import warnings
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, r2_score, confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import NearMiss 
from imblearn.over_sampling import SMOTE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE
from sklearn.model_selection import LeaveOneOut, LeavePOut, StratifiedKFold, StratifiedGroupKFold


# # Dataset

# In[ ]:


df = pd.DataFrame(pd.read_csv("global.csv"))
df = df.drop(['date', 'hour'], axis = 1)
df


# In[ ]:


feat = ['name','year','vs0', 'PSLV_v2', 'PSLV_v3', 'PSLV_v4', 'PSLV_v5', 'PSLV_v6', 'PSLV_v7',   # set of all 121 features 
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


# List of 27 features obtained from RANDOM FOREST REGRESSOR

atlantic_27_rf = ['vs0',        'DTL_t24',     'DELV-12',    'CFLX_t24',   'SHGC_t24',    
                  'SHDC_t24',    'TWXC_t24',    'TWAC_t24',    'V500_t24',    'TLON_t24',    
                  'CD20_t24',    'VMPI_t24',    'V000_t24',    'IR00_v3',     'CD26_t24',    
                  'SHRG_t24',    'MTPW_v3',     'TLAT_t24',    'PSLV_v5',     'MTPW_v10',   
                  'COHC_t24',    'VVAC_t24',    'RHHI_t24',    'EPOS_t24',    'U20C_t24',    
                  'E000_t24',    'V300_t24',    'dvs24']


# In[ ]:


df1 = pd.DataFrame(df.iloc[44660:55370], columns = feat) #Atlantic ocean (1982-2017) #change columns to atlantic_27_rf if desired
df1


# In[ ]:


df1.describe()


# # EDA and outlier removal

# In[ ]:


n_rows = 20  
n_cols = 6  

feat_toplot = [feature for feature in feat if feature not in ['name', 'year', 'dvs24', 'PSLV_v9']]
fig, axes = plt.subplots(n_rows, n_cols, figsize = (40, 100))

for i, feature in enumerate(feat_toplot):
    row = i // n_cols
    col = i % n_cols
    plot = sns.histplot(df1[feature], ax = axes[row, col], kde = True)
    plot.set_xlabel(feature, fontsize = 20)
    plot.set_ylabel('Count', fontsize = 20)
    #axes[row, col].set_title(feature)

plt.tight_layout()
plt.show()


# In[ ]:


feat_toplot = [feature for feature in feat if feature not in ['name', 'year', 'dvs24', 'PSLV_v9']]
fig, axes = plt.subplots(n_rows, n_cols, figsize=(40, 100))

for i, feature in enumerate(feat_toplot):
    row = i // n_cols
    col = i % n_cols
    ax = sns.boxplot(data=df1, x = feature, ax = axes[row, col], whis = 3.0)
    ax.set_xlabel(feature, fontsize=20)
    ax.set_ylabel('Value', fontsize=20)
    # Optionally, set title for each box plot
    # ax.set_title(f'Box Plot of {feature}', fontsize=20)

plt.tight_layout()
plt.show()


# In[ ]:


import statsmodels.api as sm
import scipy.stats as stats

feat_toplot = [feature for feature in feat if feature not in ['name', 'year', 'dvs24', 'PSLV_v9']]
fig, axes = plt.subplots(n_rows, n_cols, figsize = (40, 100))

for i, feature in enumerate(feat_toplot):
    row = i // n_cols
    col = i % n_cols
    
    # Create a QQ plot for the current feature
    sm.qqplot(df1[feature], line = 'q', ax = axes[row, col])
    
    axes[row, col].set_title(f'QQ Plot of {feature}', fontsize = 20)
    axes[row, col].set_xlabel('Theoretical Quantiles', fontsize = 16)
    axes[row, col].set_ylabel('Sample Quantiles', fontsize = 16)

plt.tight_layout()
plt.show()


# OUTLIER REMOVAL BASED ON IQR METHOD 

# In[ ]:


features_to_check = [feature for feature in feat if feature not in ['name', 'year', 'DELV-12', 'dvs24']] 

iqr_multiplier = 1.5  
outliers_dict = {}

for feature in features_to_check:
    Q1 = df1[feature].quantile(0.25)
    Q3 = df1[feature].quantile(0.75)
    
    IQR = Q3 - Q1

    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    outliers = (df1[feature] < lower_bound) | (df1[feature] > upper_bound)
    
    outliers_dict[feature] = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers_count': outliers.sum(),
        'outliers_indices': df1.index[outliers].tolist()
    }

for feature, info in outliers_dict.items():
    print(f"Feature: {feature}")
    print(f"Lower Bound: {info['lower_bound']}")
    print(f"Upper Bound: {info['upper_bound']}")
    print(f"Number of Outliers: {info['outliers_count']}")
    print('\n')


# In[ ]:


df_cleaned = df1.copy()

for feature, info in outliers_dict.items():
    outlier_indices = info['outliers_indices']
    valid_indices = [idx for idx in outlier_indices if idx in df_cleaned.index]
    df_cleaned = df_cleaned.drop(valid_indices, axis = 0)
df_cleaned.reset_index(drop = True, inplace = True)
#df_cleaned = df_cleaned.drop(['name'], axis = 1)


# In[ ]:


df_cleaned    # dataset after removing outliers from all the features


# # Feature selection

# FEATURE SELECTION BASED ON RANDOM FOREST CLASSIFIER

# In[ ]:


start = time.time()

x = df_cleaned.drop(['dvs24', 'name', 'year'], axis = 1)
y = df_cleaned['dvs24'].apply(lambda x: 1 if x >= 30 else 0)    # Target feature (encoding)

rf = RandomForestClassifier(n_estimators = 500, random_state = 1)
rf.fit(x, y)

feature_list = list(x.columns)
feature_imp = pd.Series(rf.feature_importances_, index = feature_list).sort_values(ascending = False)
print(feature_imp)

end = time.time()
print('Time:', (end - start) / 60)


# In[ ]:


plt.figure(figsize = (14, 6))
plt.bar(feature_imp.index, feature_imp.values)
plt.xticks(rotation = 90, fontsize = 8)
#plt.yticks(np.arange(0.0, 0.1, 0.01), fontsize = 12)
#plt.ylim(0.0, 0.1)
plt.xlabel('Features', fontsize = 14)
plt.ylabel('Feature Importance', fontsize = 14)
plt.title('Feature Importances', fontsize = 12)
plt.margins(x = 0.02)
plt.tight_layout()
plt.show()


# In[ ]:


list(feature_imp.index)


# # CLASSIFICATION MODEL BASED ON LOYO

# LEAVE-ONE-YEAR-OUT IMPLEMENTATION

# In[ ]:


years = df1['year'].unique()  # Get unique years from the 'year' column
loo = LeaveOneOut()

train_test_data = []  # List to store training and testing datasets

for train_index, test_index in loo.split(years):
    train_years = years[train_index]  # Years used for training
    test_year = years[test_index]  # Year used for testing

    train_data = df_cleaned[df_cleaned['year'].isin(train_years)]  # Training data
    test_data = df_cleaned[df_cleaned['year'].isin(test_year)]  # Testing data

    train_data = train_data.drop(['name', 'year'], axis = 1)  
    test_data = test_data.drop(['name', 'year'], axis = 1)

    train_test_data.append((train_data, test_data)) 


# In[ ]:


list_f1, list_pod, list_far  = [], [], []

for i in range(len(train_test_data)):
    train_data, test_data = train_test_data[i]

    # Access the training and testing datasets for the current iteration
    X_train_initial = train_data[df_cleaned.columns[2:]].values  # change between 'feat', df_cleaned.columns[2:], and 'atlantic_27_rf'
    X_train = np.delete(X_train_initial, list(df_cleaned.columns[2:]).index('dvs24'), axis = 1)  # change between...
    y_train = train_data['dvs24'].values  

    X_test_initial = test_data[df_cleaned.columns[2:]].values    # dont use `.index('dvs24')` here just the name feature set
    X_test = np.delete(X_test_initial, list(df_cleaned.columns[2:]).index('dvs24'), axis = 1)   # change between...
    y_test = test_data['dvs24'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # input data scaling
    

    
    # target feature (dvs24) encoding to 0 and 1
   
    for j in range(len(y_train)):  # target label encoding
        if y_train[j] >= 30:
            y_train[j] = 1
        else:
            y_train[j] = 0

    unri = np.where(y_train < 1)
    ri = np.where(y_train > 0)
    a, b = y_train[unri], y_train[ri]

    X_test_scaled = scaler.fit_transform(X_test)  # input data scaling

    for k in range(len(y_test)):  # target label encoding
        if y_test[k] >= 30:
            y_test[k] = 1
        else:
            y_test[k] = 0
            
            
    # SMOTE 

    sm = SMOTE(random_state = 42, 
               sampling_strategy = {0: len(a), 1: round(0.60 * len(a))}, # Input the no. of UNRI and RI samples based on sampling ratio
               k_neighbors = 10)                        
    xs, ys = sm.fit_resample(X_train_scaled, y_train)

    # SVC 
    
    svc = SVC(kernel = 'rbf', gamma = 0.001, C = 5, verbose = False)    # change SVC parameters here...
    classifier = svc.fit(xs, ys)
    ypredict = classifier.predict(X_test_scaled)
     
    #print(classification_report(y_test, ypredict))
    f1 = f1_score(y_test, ypredict)
    cm = confusion_matrix(y_test, ypredict)
    if cm.shape == (1, 1):  # Handle the case where the confusion matrix has size 1x1
        tp = cm[0, 0]
        tn, fp, fn = 0, 0, 0  # Set other values to 0, as they don't exist in this case
    else:
        tp, tn = cm[1, 1], cm[0, 0]
        fp, fn = cm[0, 1], cm[1, 0]
        
    #tp, tn = cm[1][1], cm[0][0]
    #fp, fn = cm[0][1], cm[1][0]
    pod = tp / (tp + fn)
    far = fp / (tp + fp)
    list_f1.append(f1)
    list_pod.append(pod)
    list_far.append(far)
    print('Model',i + 1, 'F1 score:', f1, 'POD:', pod, 'FAR:', far)


# In[ ]:


new_list_pod = [item for item in list_pod if not(math.isnan(item)) == True]
new_list_far = [item for item in list_far if not(math.isnan(item)) == True]
#print('Average POD:', sum(new_list_pod) / len(new_list_pod))    # average POD
#print('Average FAR:', sum(new_list_far) / len(new_list_far))    # avergae FAR
print('Average F1:', sum(list_f1) / len(list_f1))


# In[ ]:





# In[ ]:





# In[ ]:


list(df_cleaned.columns[2:])


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# # ITERATIVELY ADDING FEATURES TO MODEL TO ASSESS PERFORMANCE

# In[ ]:


start = time.time()

#sorted_rf_features = sorted_rf_feature_importance.index[:-2]  # 'PSLV_V9' and the target 'dvs24' are excluded!!!

list_f1, list_pod, list_far = [], [], []
avg_pod_list, avg_far_list, avg_f1_list = [], [], []

features_forthis = list(feature_imp.index)   # Change this anytime a different set of features is to be tested!

for i in range(len(features_forthis)):
    selected_features = features_forthis[: i + 1]
    print('Features:', selected_features)
    
    years = df1['year'].unique()
    loo = LeaveOneOut()
    train_test_data = [] 
    for train_index, test_index in loo.split(years):
        train_years = years[train_index]  
        test_year = years[test_index]  
        train_data = df_cleaned[df_cleaned['year'].isin(train_years)]  
        test_data = df_cleaned[df_cleaned['year'].isin(test_year)]  
        train_data = train_data.drop(['name', 'year'], axis = 1)  
        test_data = test_data.drop(['name', 'year'], axis = 1)
        train_test_data.append((train_data, test_data)) 

    for j in range(len(train_test_data)):
        train_data, test_data = train_test_data[j]

        X_train = train_data[selected_features].values
        y_train = train_data['dvs24'].values

        X_test = test_data[selected_features].values
        y_test = test_data['dvs24'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train) 

        for k in range(len(y_train)):  
            if y_train[k] >= 30:
                y_train[k] = 1
            else:
                y_train[k] = 0

        unri = np.where(y_train < 1)
        ri = np.where(y_train > 0)
        a, b = y_train[unri], y_train[ri]

        X_test_scaled = scaler.fit_transform(X_test)  
        for k in range(len(y_test)):  
            if y_test[k] >= 30:
                y_test[k] = 1
            else:
                y_test[k] = 0

        sm = SMOTE(random_state = 42, sampling_strategy = {0: len(a), 1: round(0.60 * len(a))}, k_neighbors = 10) # can change SMOTE RATIO                      
        xs, ys = sm.fit_resample(X_train_scaled, y_train)

        svc = SVC(kernel = 'rbf', gamma = 0.005, C = 10, verbose = False)   # change SVC parameters here...
        classifier = svc.fit(xs, ys)
        ypredict = classifier.predict(X_test_scaled)

        f1 = f1_score(y_test, ypredict)
        cm = confusion_matrix(y_test, ypredict)
        
        if cm.shape == (1, 1):  # Handle the case where the confusion matrix has size 1x1
            tp = cm[0, 0]
            tn, fp, fn = 0, 0, 0  # Set other values to 0, as they don't exist in this case
        else:
            tp, tn = cm[1, 1], cm[0, 0]
            fp, fn = cm[0, 1], cm[1, 0]
        
        #tp, tn = cm[1][1], cm[0][0]
        #fp, fn = cm[0][1], cm[1][0]
        pod = tp / (tp + fn)
        far = fp / (tp + fp)
        list_f1.append(f1)
        list_pod.append(pod)
        list_far.append(far)
        
    avg_f1 = np.nanmean(np.array(list_f1))
    avg_pod = np.nanmean(np.array(list_pod))
    avg_far = np.nanmean(np.array(list_far))
    
    if i >= 40 and avg_f1 <= 0.2800:
        break
    
    avg_f1_list.append(avg_f1)
    avg_pod_list.append(avg_pod)
    avg_far_list.append(avg_far)

    print(f"Number of features: {i + 1}")
    print(f"Average F1 score: {avg_f1}")
    #print(f"Average POD: {avg_pod}")
    #print(f"Average FAR: {avg_far}")
    print()
    
end = time.time()
print('time elapsed: ', (end - start) / 60, 'min')    


# In[ ]:





# # HYPERPARAMETER TUNING 

# In[ ]:


start = time.time()

list_f1, list_pod, list_far = [], [], []
avg_pod_list, avg_far_list, avg_f1_list = [], [], []

features_forthis = list(feature_imp.index)  

gamma = [0.0005, 0.001, 0.005, 0.01]
cost = [5, 10, 15, 20]

count, list_avg_f1 = 0, []

#for rat in range(len(ratio)):               # CURRENTLY for LOOPS HAVE BEEN USED FOR ITERATING THROUGH  PARAMETER COMBINATIONS 
#    print('Sampling ratio:', ratio[rat])
for gm in range(len(gamma)):
    print('Gamma:', gamma[gm])
    for c in range(len(cost)):
        print('Cost:', cost[c])
        print()

        for i in range(len(features_forthis)):
            selected_features = features_forthis[: i + 1]
            #print('Features:', selected_features)

            years = df1['year'].unique()
            loo = LeaveOneOut()
            train_test_data = [] 
            for train_index, test_index in loo.split(years):
                train_years = years[train_index]  
                test_year = years[test_index]  
                train_data = df_cleaned[df_cleaned['year'].isin(train_years)]  
                test_data = df_cleaned[df_cleaned['year'].isin(test_year)]  
                train_data = train_data.drop(['name', 'year'], axis = 1)  
                test_data = test_data.drop(['name', 'year'], axis = 1)
                train_test_data.append((train_data, test_data)) 

            for j in range(len(train_test_data)):
                train_data, test_data = train_test_data[j]

                X_train = train_data[selected_features].values
                y_train = train_data['dvs24'].values

                X_test = test_data[selected_features].values
                y_test = test_data['dvs24'].values

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train) 

                for k in range(len(y_train)):  
                    if y_train[k] >= 30:
                        y_train[k] = 1
                    else:
                        y_train[k] = 0

                unri = np.where(y_train < 1)
                ri = np.where(y_train > 0)
                a, b = y_train[unri], y_train[ri]

                X_test_scaled = scaler.fit_transform(X_test)  
                for k in range(len(y_test)):  
                    if y_test[k] >= 30:
                        y_test[k] = 1
                    else:
                        y_test[k] = 0

                sm = SMOTE(random_state = 42, sampling_strategy = {0: len(a), 1: round(0.60 * len(a))}, k_neighbors = 10)                        
                xs, ys = sm.fit_resample(X_train_scaled, y_train)

                svc = SVC(kernel = 'rbf', gamma = gamma[gm], C = cost[c], verbose = False)
                classifier = svc.fit(xs, ys)
                ypredict = classifier.predict(X_test_scaled)

                f1 = f1_score(y_test, ypredict)
                cm = confusion_matrix(y_test, ypredict)
                list_f1.append(f1)

            avg_f1 = np.nanmean(np.array(list_f1))

            if i >= 40 and avg_f1 <= 0.2800:
                break

            avg_f1_list.append(avg_f1)

            print(f"Features: {i + 1}  Avg. F1: {avg_f1}")
            #print(f"Average F1 score: {avg_f1}")
        print()
    
end = time.time()
print('time elapsed: ', (end - start) / 60, 'min')    


# In[ ]:


plt.figure(figsize = (8, 5))
plt.plot(avg_f1_list, 'o-')
plt.xlabel('Number of features', fontsize = 14)
plt.ylabel('F1 score', fontsize = 14)
#plt.yticks(np.arange(0.16, 0.28, 0.02))
#plt.ylim(0.15, 0.27)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




