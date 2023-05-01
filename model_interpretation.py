#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import numpy as np
import pandas as pd
import shap
import types
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss 
from imblearn.over_sampling import SMOTE
from explainerdashboard import ClassifierExplainer, ExplainerDashboard 


# LOAD DATA

# In[4]:


df = pd.DataFrame(pd.read_csv("global.csv"))
df = df.drop(['date', 'hour'], axis = 1)
df


# In[5]:


# List of retained features for each basin after random forest ranking 

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


# In[39]:


# Training data (df1)

df1 = pd.DataFrame(df.iloc[44660:54960], columns = atlantic_22_rf)  # Atlantic ocean (upto 2016) 
#df1 = pd.DataFrame(df.iloc[26616:44204], columns = feat)            # Western pacific (upto 2016)
#df1 = pd.DataFrame(df.iloc[13399:15547], columns = indian_10_rf)    # Indian Ocean (upto 2014)
#df1 = pd.DataFrame(df.iloc[13399:15729], columns = indian_10_rf)    # Indian Ocean (upto 2016)
#df1 = pd.DataFrame(df.iloc[: 13062], columns = east_36_rf)          # Eastern Pacific (upto 2016) 
df1


# In[40]:


# Testing data (df3)

df3 = pd.DataFrame(df.iloc[54960:55370], columns = atlantic_22_rf)   # AL 2017
#df3 = pd.DataFrame(df.iloc[44204:44662], columns = feat)   # WP 2017
#df3 = pd.DataFrame(df.iloc[15547:15778], columns = indian_10_rf)   # IO 2015-2017
#df3 = pd.DataFrame(df.iloc[13062:13397], columns = east_36_rf)   # EP 2017
df3


# DATA PRE-PROCESSING AND SCALING

# In[41]:


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


# SMOTE AND SVM TRAINING

# In[42]:


#SMOTE for handling class imbalance
sm = SMOTE(random_state = 42, sampling_strategy = {0: 9768, 1: 2442}, k_neighbors = 10)  # Sampling strategy dict:      
xs, ys = sm.fit_resample(xtrain, ytrain)                                                 # Atlantic: {0: 9768, 1: 2442}  
                                                                                         # Indian: {0: 2020, 1: 2020}

#SVM training and fitting
svc = SVC(kernel = 'rbf', gamma = 0.002, C = 15, verbose = 2)   # gamma = 0.002 for Atlantic and 0.005 for Indian (same C).
classifier = svc.fit(xs, ys)


# MODEL INTERPRETATION USING EXPLAINER DASHBOARD

# In[ ]:


start = time.time()

def predict_proba(self, X):
    pred = self.predict(X)
    return np.array([1-pred, pred]).T
svc.predict_proba = types.MethodType(predict_proba, svc)

xtrain_df, xtest_df = pd.DataFrame(xtrain, columns = cols), pd.DataFrame(xtest, columns = cols)
background_df = xtrain_df.sample(n = 100, random_state = 42) # Change background dataset for slower or faster SHAP computation  
explainer = ClassifierExplainer(classifier, xtest_df, ytest, X_background = background_df)  

db = ExplainerDashboard(explainer, model_summary = True, shap_interaction = True,
                        shap_dependance = True, dpi = 600)

db.run(mode = 'inline')   

end = time.time()
print('time elapsed:', (end - start) / 60, 'min')


# In[38]:


fig = explainer.plot_dependence("TWXC", color_col = "COHC_t24")   # Change the feature names to see their dependence

fig.update_layout(
    height = 350,
    width = 450,
    xaxis = dict(
        title = "TWXC_t24",
        title_font = dict(size = 14),
        title_standoff = 0,
        tickfont = dict(size = 14),
        showline = False,
    ),
    yaxis = dict(
        title = "SHAP value for TWXC_t24",
        title_font = dict(size = 14),
        title_standoff = 0,
        tickfont = dict(size = 14),
        showline = False,
    ),
)


# MODEL INTERPRETATION USING SHAP LIBRARY

# In[ ]:


import shap

background = shap.kmeans(xs, 100)   # second argument represents the background data samples
explainer2 = shap.LinearExplainer(classifier.decision_function, background)
shap_values = explainer2.shap_values(xs)


# In[ ]:


shap_values_unscaled = scaler.inverse_transform(shap_values)   # Convert scaled SHAP values to unscaled 


# In[ ]:


fig, ax = plt.subplots(figsize = (6, 4.5))
shap.dependence_plot(6, shap_values_unscaled, df_test,
                     interaction_index = 'auto',
                     ax = ax, show = False)       # list(df_test.columns).index(most_correlated)

ax.set_xlabel("TWXC_t24", fontsize = 14, labelpad = 8)        
ax.set_ylabel("SHAP Value for TWXC_t24", fontsize = 14, labelpad = 8) 
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
ax.set_yticks(np.arange(50, 500, 100))
ax.set_xticks(np.arange(50, 500, 100))
ax.tick_params(axis = 'both', which='both', direction = 'in', width = 1, length = 5, labelsize = 12)
ax.tick_params(labelsize = 12)

plt.show()

