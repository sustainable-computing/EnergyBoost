import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns

hourtable = pd.read_csv('price_final.csv')['price'][0:30*24]
csvfile = open('price_plot.csv','w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(["Hour","price"])
for i in range(30*24):
    writer.writerow([i%24,hourtable.iloc[i]])
csvfile.close()


pricetable = pd.read_csv("price_plot.csv")
hour = np.arange(24)
price_tou = np.array([0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.132, 0.132, 0.132, 0.132, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.132, 0.132, 0.065, 0.065, 0.065, 0.065, 0.065])
price_tou = price_tou/1.3
ax = sns.lineplot(x="Hour", y="price", data=pricetable)
#ax.plot(hour, price_tou, label="RBC")
plt.show()
