import pandas as pd
import csv
def create():
    """
    create hourly price table,
    read the the raw data with unix time
    convert to utc time with proper time zone

    :return: None

    """
    df = pd.read_csv('/home/baihong/Documents/data/price.csv')
    csvfile = open("bill/price.csv",'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["time","price"])

    df['newtime']=pd.DatetimeIndex(pd.to_datetime(df['time'], unit='ms'))\
                     .tz_localize('UTC' )\
                     .tz_convert('US/Central')




    for i in range((df.shape[0]-1),0,-12):
        #print(df.iloc[[i]])
        price = df['price'].iloc[i]
        price= round(price*0.01,3)
        writer.writerow([price])


    csvfile.close()
