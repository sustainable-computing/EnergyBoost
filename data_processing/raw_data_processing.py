
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[2]:


solar_df = pd.read_csv('data_all/Solar.csv')
print(solar_df.shape)
solar_df.head(10)


# In[3]:


hh_df = pd.read_csv('data_all/all_data_raw.csv')
print(hh_df.shape)
hh_df.head(15)


# In[4]:


hh_df = hh_df[['dataid', 'localhour', 'use', 'temperature', 'cloud_cover']]
hh_df.head()


# In[5]:


unique_dataid = np.unique(hh_df['dataid'])
hhids = list(unique_dataid)
#hhids = [26, 59, 77, 86, 93, 94, 101, 114, 115, 171, 187, 252, 330, 370, 379, 410, 434, 483, 484, 499, 503, 545, 585, 621, 624, 661, 668, 744, 781, 796, 871, 890, 930, 946, 974, 994, 1086, 1103, 1169, 1185, 1192, 1202, 1283, 1310, 1334, 1354, 1403, 1415, 1463, 1500, 1507, 1551, 1632, 1642, 1697, 1700, 1714, 1718, 1790, 1792, 1796, 1800, 1801, 1947, 1953]
hhids = [86,93] 
print(hhids)


# In[6]:


df_by_hh = {}
for hhid in hhids:
    df_by_hh[hhid] = []


# In[7]:


solar_df.columns = ['date', 'localhour', 'GH']
solar_df.head()


# In[8]:


time_gh = {}


# In[9]:


for index, row in solar_df.iterrows():
    date = row.date.split('/')
    if len(date[0]) < 2:
        date[0] = '0' + date[0]
    if len(date[1]) < 2:
        date[1] = '0' + date[1]
    time_str = '{}-{}-{} {}'.format(date[-1], date[0], date[1], row.localhour)
    time_gh[time_str] = row.GH


# In[10]:


len(time_gh)


# In[11]:


hh_df.localhour[:10]


# In[12]:


hh_df['GH'] = 0.0
hh_df['is_weekday'] = 0
hh_df['month'] = 0.0
hh_df['hour'] = 0.0
hh_df.iloc[-50000]


# In[13]:


used = ['localhour', 'use', 'temperature', 'cloud_cover','GH', 'is_weekday','month','hour']


# In[14]:


for index, row in hh_df.iterrows():
    row.month = float(pd.to_datetime(row.localhour[:-3]).month)
    row.hour = float(pd.to_datetime(row.localhour[:-3]).hour)
    try:
        row.GH = time_gh[row.localhour[:-6]]
        if row.localhour[-1] == '5':
            row.is_weekday = int(datetime.strptime(str(row.localhour), "%Y-%m-%d %H:%M:%S-05").weekday() < 5)
        else:
            row.is_weekday = int(datetime.strptime(str(row.localhour), "%Y-%m-%d %H:%M:%S-06").weekday() < 5)
        df_by_hh[row.dataid].append(row[used])
    except:
        print(row.localhour[:-6])


# In[15]:


for i in hhids:
    print('\nProcessing data of household {}, {} samples in total.'.format(i, len(df_by_hh[i])))
    df = pd.DataFrame(data=df_by_hh[i], columns=used)
    print(df.head())
    df.to_csv('data2016/processed_hhdata_{}.csv'.format(i))

