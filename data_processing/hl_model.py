
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
from sklearn.externals import joblib

from sklearn import ensemble, tree

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from __future__ import division, print_function

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels     import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

print(__doc__)

import warnings
warnings.filterwarnings("ignore")


# In[2]:


hhids=[86]
#hhids=[26, 59, 77, 86, 93, 94, 101, 114, 171, 187]
#hhids=[2532, 2557, 2575, 2638, 2641, 2742, 2750, 2755, 2769, 2787, 2814, 2818, 2829, 2859, 2925, 2945, 2953, 2965, 2980, 2986]
#for filename in glob.glob('data_added2/added_hhdata_*'):
    #basename=filename.split("/")[1].split(".")[0]
    #hhid=basename.split("_")[2]
for hhid in hhids:
    print('Start :: Process on household {}...'.format(hhid))
    df = pd.read_csv('data_added2/added_hhdata_{}_2.csv'.format(hhid), index_col=0)
    df = df.dropna()
    st = []
    ct = 0
    for idx, row in df.iterrows():
        if row.GH < 2000 and row.GH > -1000:
            st.append(row)
        else:
            ct += 1
    print(ct)

    df = pd.DataFrame(data=st, columns=df.columns)
    #features = [ 'GH', 'use_hour','use_week', 'temperature', 'cloud_cover','wind_speed','is_weekday','month','hour']
    features = ['temperature', 'is_weekday','month','hour']


    
    Y = list(df.use)[1:]
    try:
        Y.append(df.use.iloc[0])
    except:
        break
    Y = np.array(Y)

    X = df[features]
    X = np.array(X)
    print(X.shape)
    print(Y.shape)

    temp_df = pd.DataFrame(data=X, columns=features)
    temp_df = temp_df[features]
    temp_df['y_use'] = Y
    values = temp_df.values

    # normalize features
    scaler = MinMaxScaler()
    y_gt = values[:,-1:]
    scaled = scaler.fit_transform(values)
    values = scaled

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.10,
                                                        random_state=666)
    
    


    clf = tree.DecisionTreeRegressor()

    clf.fit(X_train, Y_train)
    yhat = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, yhat))
    mae = mean_absolute_error(Y_test, yhat)
    print("Decision Tree")
    print('RMSE =>', rmse)
    print('MAE =>', mae)
    
    # Kernel with parameters given in GPML book
    k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
    k2 = 2.4**2 * RBF(length_scale=90.0)         * ExpSineSquared(length_scale=1.3, periodicity=24.0*7)  # seasonal component
    # medium term irregularity
    k3 = 0.66**2         * RationalQuadratic(length_scale=1.2, alpha=0.78)
    k4 = 0.18**2 * RBF(length_scale=0.134)         + WhiteKernel(noise_level=0.19**2)  # noise terms
    kernel_gpml = k1 + k2 + k3 + k4

    gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=5000,
                                  optimizer=None, normalize_y=True)
    gp.fit(X_train, Y_train)
    yhat, y_std = gp.predict(X_test, return_std=True)
    rmse = np.sqrt(mean_squared_error(Y_test, yhat))
    mae = mean_absolute_error(Y_test, yhat)
    print("GPR")
    print('RMSE =>', rmse)
    print('MAE =>', mae)
    

#     print("GPML kernel: %s" % gp.kernel_)
#     print("Log-marginal-likelihood: %.3f"
#           % gp.log_marginal_likelihood(gp.kernel_.theta))
    
    #joblib.dump(clf, 'saved_models/hl_rf_{}.pkl'.format(hhid))


# In[ ]:


#new_clf = joblib.load('saved_models/hl_rf_86.pkl') 


# In[ ]:


#X_train[0:1]


# In[ ]:


#new_clf.predict(X_train[0:1])
#np.savetxt("pre_86.csv", new_clf.predict(X_train[0:1]), delimiter=",")


# In[ ]:


#Y_train[0:1]
#np.savetxt("real_86.csv",Y_train[0:1] , delimiter=",")

