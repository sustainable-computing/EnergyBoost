
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[2]:


hh_df = pd.read_csv('home_ac/processed_hhdata_86_2.csv')
# print(hh_df.shape)
# hh_df.head(15)
hh_df.drop_duplicates(subset ="localhour", keep = False, inplace = True)
print(hh_df.shape)


# In[3]:


hh_df['hour_index']=0
#hh_df.iloc[-50]


# In[4]:


used = ['localhour', 'use', 'temperature', 'cloud_cover','GH', 'is_weekday','month','hour','AC','DC','hour_index']
datarow= []


# In[5]:



hour_index=0#hour index
hour_value=0
missing_count=0
start_time= pd.to_datetime(hh_df['localhour'].iloc[0][:-3])
for index, row in hh_df.iterrows():
    row.localhour=row.localhour[:-3]
    #print(row.localhour)
    difference=(pd.to_datetime(row.localhour)-pd.to_datetime(hh_df['localhour'].iloc[0][:-3])).total_seconds()/3600
    #print("index is difference",difference)
    
    if difference!=hour_index:
        gap = difference-hour_index
        missing_count += gap
        #fill in the missing hours
        for i in range(int(gap)):
            print("\n---------------------------------------")
            print("missing data for hour index:",hour_index+i)
            #row.hour=(hour_index+i)%24
            temprow=None
            #print("this is lastrow",lastrow)
            temprow=lastrow
            #print("this is temprow",temprow)
            temprow.hour_index=hour_index+i
            #print("this is hour of lastrow",lastrow.hour)
            #temprow.hour = (hour_index+i)%24
            current_time = start_time+pd.Timedelta(hour_index+i,unit='h')
            temprow.localhour = current_time
            temprow.hour = current_time.hour
            temprow.month = current_time.month
            temprow.is_weekday = int(datetime.strptime(str(current_time), "%Y-%m-%d %H:%M:%S").weekday() < 5)
            print("The inserted row is \n",temprow)
            #datarow.append(row[used])
            datarow.append(temprow[used])
            temprow=None
            #hour=None
            #print(datarow)
        hour_index = difference
    hour_index +=1
    row.hour_index=difference
    #hour_value = row.hour
    #print(row[used])
    #print("reach here")
    lastrow = row[used]
    datarow.append(row[used])
print("total missing hours",missing_count)



    
       
#------------------------------------------testing----------------------------
# hour_index=0 #hour index
# missing_count=0
# for index, row in hh_df.iterrows():
#     #print(row.localhour)
#     #row.month = float(pd.to_datetime(row.localhour[:-3]).month)
#     #row.day = float(pd.to_datetime(row.localhour[:-3]).day)
#     #data_hour = float(pd.to_datetime(row.localhour).hour-6)%24
#     data_hour = float(pd.to_datetime(row.localhour[:-3]).hour)
#     #print(data_hour)
#     if data_hour != hour_index%24:
#         print("we are missing hours for",row.localhour)
#         missing_count += 1
#         hour_index +=1
#     hour_index += 1
# print("In total missing hours", missing_count)

# for index, row in hh_df.iterrows():
#     #row.month = float(pd.to_datetime(row.localhour[:-3]).month)
#     #row.day = float(pd.to_datetime(row.localhour[:-3]).day)
#     print("------------")
#     print(row.localhour)
#     print(float(pd.to_datetime(row.localhour).hour-6)%24)
#     print(float(pd.to_datetime(row.localhour[:-3]).hour))
# #     print(pd.to_datetime(row.localhour))
# #     print(pd.to_datetime(row.localhour).tz_localize('UTC'))
# #     print(pd.to_datetime(row.localhour).tz_localize('UTC').tz_convert('US/Central'))
# #     print(pd.to_datetime(row.localhour[:-3]).tz_localize('US/Central'))
# #     print(pd.to_datetime(row.localhour)-pd.Timedelta('06:00:00'))



# In[6]:


df = pd.DataFrame(data=datarow, columns=used)
print(df.head())
df.to_csv('datanew/afterfix6.csv')

