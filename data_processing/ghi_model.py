
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
from sklearn.externals import joblib

from sklearn import ensemble

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#hhids=[26, 59, 77, 86, 93, 94, 101, 114, 115, 160, 171, 187, 222, 252, 370, 379, 410, 434, 483, 484, 499, 503, 545, 575, 580, 585, 621, 624, 661, 668, 739, 744, 781, 796, 871, 890, 898, 930, 936, 946, 974, 994, 1086, 1103, 1169, 1185, 1192, 1202, 1283, 1310, 1314, 1334, 1354, 1403, 1415, 1463, 1500, 1507, 1551, 1589, 1617, 1632, 1642, 1681, 1697, 1700, 1714, 1718, 1790, 1791, 1792, 1796, 1800, 1801, 1947, 1953, 2004, 2018, 2034, 2072, 2094, 2129, 2156, 2158, 2171, 2199, 2204, 2233, 2242, 2335, 2337, 2361, 2365, 2378, 2401, 2449, 2461, 2470, 2472, 2510, 2532, 2557, 2575, 2638, 2641, 2742, 2750, 2755, 2769, 2787, 2814, 2818, 2829, 2859, 2925, 2945, 2953, 2965, 2980, 2986]


for filename in glob.glob('data_added2/added_hhdata_*'):
    basename=filename.split("/")[1].split(".")[0]
    hhid=basename.split("_")[2]
    #result[hhid] = []
    print('Start :: Process on household {}...'.format(hhid))
    df = pd.read_csv('data_added2/added_hhdata_{}_2.csv'.format(hhid), index_col=0)
    st = []
    ct = 0
    for idx, row in df.iterrows():
        if row.GH < 2000 and row.GH > -1000:
            st.append(row)
        else:
            ct += 1

    # print(ct)
    df = pd.DataFrame(data=st, columns=df.columns)
    features = ['use', 'temperature', 'cloud_cover','wind_speed','is_weekday','ac_hour','ac_week','month','hour']



    Y = list(df.ac)[1:]
    Y.append(df.ac.iloc[0])
    Y = np.array(Y)

    X = df[features]
    X = np.array(X)
    X.shape

    temp_df = pd.DataFrame(data=X, columns=features)
    temp_df['y_GH'] = Y
    values = temp_df.values

    # normalize features
    scaler = MinMaxScaler()
    y_gt = values[:,-1:]
    scaled = scaler.fit_transform(values)
    values = scaled

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.10,
                                                        random_state=666)


    clf = ensemble.RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=75, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

    clf.fit(X_train, Y_train)
    yhat = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, yhat))
    mae = mean_absolute_error(Y_test, yhat)
    print('RMSE =>', rmse)
    print('MAE =>', mae)
    joblib.dump(clf, 'saved_models/ghi_rf_{}.pkl'.format(hhid))


# In[3]:


new_clf = joblib.load('saved_models/ghi_rf_26.pkl') 


# In[4]:


X_train[0:2]


# In[5]:


new_clf.predict(X_train[0:2])


# In[6]:


Y_train[0:2]

