
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


input_file = pd.read_csv('home_ac/processed_hhdata_86_2.csv')[0:8640][['use','AC']]
input_file.head(10)


# In[3]:


input_result = pd.read_csv('86_2/sb20b64.csv')[['Base_Soc','Base_Action']]
input_result.head(10)


# In[4]:


#training set of 8640
pair= pd.concat([input_file,input_result],axis=1)
pair.head(20)


# In[5]:


#sample 1000 from training set
#The Matr
sample=pair.sample(200)
#print(sample)
con1=sample.cov()
print(con1)


# In[6]:


def kernal(m1,m2):
    global con1
    result= np.exp (-(np.subtract(m1,m2)).transpose()*(np.subtract(m1,m2))*np.linalg.inv(con1/2))
    return result


# In[7]:


#sample['use'].iloc[0]
# for i in range(10):
#     step_m = np.matrix([[pair['use'][0],pair['AC'][0],pair['Base_Soc'][0],pair['Base_Action'][0]]])
#     sample_m = np.matrix([[sample['use'][i],sample['AC'][i],sample['Base_Soc'][i],sample['Base_Action'][i]]])
#     print(kernal(step_m,sample_m)) 


# In[8]:


all_states=[]
for i in range(2000):
    feature_v=[]
    for j in range(len(sample)):
        step_m = np.matrix([[pair['use'][i],pair['AC'][i],pair['Base_Soc'][i],pair['Base_Action'][i]]])
        sample_m = np.matrix([[sample['use'].iloc[j],sample['AC'].iloc[j],sample['Base_Soc'].iloc[j],sample['Base_Action'].iloc[j]]])
        feature_v.append(kernal(step_m,sample_m))
    all_states.append(feature_v)
all_states

