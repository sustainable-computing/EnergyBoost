
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


hhids=[26, 59, 77, 86, 93, 94, 101, 114, 171, 187]


for hhid in hhids: 
    result[hhid] = []
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

#     temp_df = pd.DataFrame(data=X, columns=features)
#     temp_df['y_GH'] = Y
#     values = temp_df.values

#     # normalize features
#     scaler = MinMaxScaler()
#     y_gt = values[:,-1:]
#     scaled = scaler.fit_transform(values)
#     values = scaled

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
        print(clf)
        yhat = clf.predict(X_test)
        scores = cross_val_score(clf, X_train, Y_train)
        rmse = np.sqrt(mean_squared_error(Y_test, yhat))
        mae = mean_absolute_error(Y_test, yhat)
        print('RMSE =>', rmse)
        print('MAE =>', mae)
        print('CV Score =>', scores)
        model_dict = {
            'name': clf.__class__.__name__,
            'rmse': rmse,
            'mae': mae,
        }
        result[hhid].append(model_dict)
        print('')


# In[4]:


final = []
for k, v in result.items():
    for i in result[k]:
        final.append([str(k), i['name'], i['rmse'], i['mae']])
col = ['household_id', 'alg', 'RMSE', 'MAE']
final = pd.DataFrame(data=final, columns=col)
final.to_csv('GHI.csv')


# In[5]:


final


# In[6]:


gb = final.groupby('alg')


# In[7]:


N = 9
ind = range(N)
mean = list(gb['RMSE'].describe()['mean'])
mean.append()
std = list(gb['RMSE'].describe()['std'])
std.append()


# In[8]:


print(mean, std)


# In[9]:


p1 = plt.bar(ind, mean, 0.4, yerr=std)

plt.ylabel('GHI ($W/m^2$)')
plt.title('Prediction RMSE of different models')
plt.xlabel('Model')
plt.xticks(ind, ('Ridge', 'Lasso','BR' ,'LasLar', 'LR', 'RF', 'DTR', 'MLP','GPR'))
# plt.yticks(np.arange(50, 2))
plt.axhline(y=gb['RMSE'].describe()['mean']['RandomForestRegressor'], linewidth=0.15)
plt.savefig('ghi.png')

