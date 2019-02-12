
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model, ensemble, svm, tree, neural_network

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

import warnings
warnings.filterwarnings("ignore")


# In[2]:


result = {}


# In[3]:


#hhids=[86, 59, 77, 26, 93, 101, 114, 171, 1086, 1403]
hhids=[26, 59, 77, 86, 93, 94, 101, 114, 171, 187]
#hhids=[86]
for hhid in hhids: 
    X=[]
    result[hhid] = []
    print('Start :: Process on household {}...'.format(hhid))
    df = pd.read_csv('data_filled2/processed_hhdata_{}_2.csv'.format(hhid), index_col=0)

    features = [ 'GH','temperature', 'cloud_cover','wind_speed','is_weekday','month','hour']
    
    Y = list(df.use)[500:]
    Y = np.array(Y)
    print(Y.shape)
    print(Y[0])
    
    #get X
    for index, row in df.iterrows():
        if index>=500:
            rowlist=row[features]
            rowlist = rowlist.tolist()
            X.append(rowlist)
            rowlist.append(df.use.iloc[index-1])
            #rowlist.append(df.use.iloc[index-2])
            #rowlist.append(df.use.iloc[index-23])
            #rowlist.append(df.use.iloc[index-24])
            rowlist.append(df.use.iloc[index-168])
            
#             for i in range(1,169):
#                 rowlist.append(df.use.iloc[index-i])
                
             
    #X = df[features]
    X = np.array(X)
    print(X.shape)
    print(X[0])
    print(len(X[0]))



    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.10,
                                                        random_state=666)


    classifiers = [
        linear_model.Ridge(alpha=1.0, random_state=0),
        linear_model.Lasso(alpha=0.55, random_state=0),
        linear_model.BayesianRidge(alpha_1=1e-06, alpha_2=1e-06),
        linear_model.LassoLars(alpha=0.55),
        linear_model.LinearRegression(),
        ensemble.RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=75, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False),
        tree.DecisionTreeRegressor(),
        neural_network.MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(50, 50), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=0, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    ]


    print('Start :: Find the best model for this household...')
    for clf in classifiers:
        clf.fit(X_train, Y_train)
#         print(clf)
        yhat = clf.predict(X_test)
        scores = cross_val_score(clf, X_train, Y_train)
        rmse = np.sqrt(mean_squared_error(Y_test, yhat))
        nrmse = rmse/(df.use.max()-df.use.min())
        mae = mean_absolute_error(Y_test, yhat)
        print('RMSE =>', rmse)
        print('nRMSE =>', nrmse)
        print('MAE =>', mae)
        print('CV Score =>', scores)
        model_dict = {
            'name': clf.__class__.__name__,
            'rmse': rmse,
            'nrmse': nrmse,
            'mae': mae,
        }
        result[hhid].append(model_dict)
#         print('')


# In[4]:


final = []
for k, v in result.items():
    for i in result[k]:
        final.append([str(k), i['name'], i['rmse'],i['nrmse'], i['mae']])
col = ['household_id', 'alg', 'RMSE','nRMSE', 'MAE']
final = pd.DataFrame(data=final, columns=col)
final.to_csv('HL.csv')


# In[5]:


final = pd.read_csv('HL.csv', index_col=0)
final


# In[6]:


gb = final.groupby('alg')


# In[7]:


N = 9
ind = range(N)
mean = list(gb['nRMSE'].describe()['mean'])
mean.append(0.12464348324161187)
std = list(gb['nRMSE'].describe()['std'])
std.append(0.04447879299698875)


# In[8]:


ind, mean, std


# In[9]:


# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax2 = ax.twinx()
# ax.set_ylabel('RMSE1')
# # ax2.set_ylabel('RMSE2')


# # ax.bar(ind, ghi_mean, 0.3, yerr=ghi_std, color='red', align='center')
# # ax.autoscale(tight=True)
# plt.show()

plt.bar(ind, mean, 0.4, yerr=std, align='center')
plt.ylabel('%')
plt.xlabel('Model')
plt.title('Prediction home use nRMSE of different models')
plt.xticks(ind, ('Ridge', 'Lasso','BR' ,'LasLar', 'LR', 'RF', 'DTR', 'MLP','GPR'))
# plt.yticks(np.arange(0, 2))
plt.axhline(y=gb['nRMSE'].describe()['mean']['RandomForestRegressor'], linewidth=0.15)
plt.savefig('hl.png')


# In[10]:


fig = plt.figure()

