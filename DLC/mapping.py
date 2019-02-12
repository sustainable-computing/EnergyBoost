
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import time

from sklearn import linear_model, ensemble, svm, tree, neural_network

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


result = {}


# In[ ]:



#hhids=[26, 59, 77, 86, 93, 94, 101, 114, 171, 187]
hhids=['1202', '871', '1103', '585', '59', '2755', '2233', '86', '114', '2575', '1700', '974', '1800',
 '370', '187', '1169', '1718', '545', '94', '2018', '744', '2859', '2925', '484', '2953', '171', '2818', '1953',
 '1697', '1463', '499', '1790', '1507', '1642', '93', '1632',
 '1500', '2472', '2072', '2378', '1415', '2986', '1403', '2945', '77', '1792',
 '624', '379', '2557', '890', '1192', '26', '2787', '2965', '2980', '434', '2829',
 '503', '2532', '946', '2401', '1801','2337','1086','1714','1283','252','2814']
scenarios=["sb4b64","sb4b135","sb8b64","sb8b135","sb10b64","sb10b135","sb20b64","sb20b135"]

for hhid in hhids:
    print("Working on home",hhid)
    for j in scenarios:
        print("Scenario:",j)
        nj=j[0:2]+'-'+j[2:]
        print('Start :: Process on household {}...'.format(hhid))
        df = pd.read_csv('data_filled4/processed_hhdata_{}_4.csv'.format(hhid), index_col=0)
        df_action = pd.read_csv('op_4/{}_2_sc/{}.csv'.format(hhid,nj), index_col=0)

        features = [ 'temperature', 'cloud_cover','wind_speed','is_weekday','month','hour']

        Y = list(df_action.Best_Action)[0:8736]
        Y = np.array(Y)
#         print(Y.shape)
#         print(Y[0])

        X = df[features][0:8736]
        X = np.array(X)
#         print(X.shape)
#         print(X[0])
#         print(len(X[0]))




        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.10,
                                                            random_state=666)

        X_test = X
        Y_test = Y


        classifiers = [
#         neural_network.MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#         beta_2=0.999, early_stopping=False, epsilon=1e-08,
#         hidden_layer_sizes=(50, 50), learning_rate='constant',
#         learning_rate_init=0.001, max_iter=200, momentum=0.9,
#         nesterovs_momentum=True, power_t=0.5, random_state=0, shuffle=True,
#         solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
#         warm_start=False)
        neural_network.MLPRegressor(hidden_layer_sizes=(100, ), 
        activation='relu', solver='adam', alpha=0.0001, batch_size='auto', 
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
        max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
        warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
        ]
        



        print('Fitting action...')
        for clf in classifiers:
            clf.fit(X_train, Y_train)
            start_time = time.time()
            yhat = clf.predict(X_test)
            elapsed_time = time.time() - start_time
            print(elapsed_time)
            #pathlib.Path("olc_4/{}_4_action".format(hhid)).mkdir(parents=True, exist_ok=True)
            #np.savetxt("olc_4/{}_4_action/{}.csv".format(hhid,j), yhat, delimiter=",")

