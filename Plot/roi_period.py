import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

#hhids=[86, 59, 77, 26, 93, 101, 114, 171, 1086, 1403]
# hhids=['1202', '871', '1103', '585', '59', '2755', '2233', '86', '114', '2575', '1700', '974', '1800',
#  '370', '187', '1169', '1718', '545', '94', '2018', '744', '2859', '2925', '484', '2953', '171', '2818', '1953',
#  '1697', '1463', '499', '1790', '1507', '1642', '93', '1632',
#  '1500', '2472', '2072', '2378', '1415', '2986', '1403', '2945', '77', '1792',
#  '624', '379', '2557', '890', '1192', '26', '2787', '2965', '2980', '434', '2829',
#  '503', '2532', '946', '2401', '1801','2337','1086','1714','1283','252','2814']


#compile
csvfile = open('violin/periodsb10b135.csv','w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(["Payback period"," ","Policy","PV system’s size"])


table=pd.read_csv('period/panel2sb10b135.csv')
for i in range(68):
    print(i)

    year =  table['No solar no incentive MPC'][0:68].iloc[i]
    writer.writerow([year,"Without incentives","MPC","4.4kWp Battery size = 13.5KWh"])

    year =  table['No solar no incentive RBC'][0:68].iloc[i]
    writer.writerow([year,"Without incentives","RBC","4.4kWp Battery size = 13.5KWh"])

    year =  table['No solar with incentive MPC'][0:68].iloc[i]
    writer.writerow([year,"With incentives","MPC","4.4kWp Battery size = 13.5KWh"])

    year =  table['No solar with incentive RBC'][0:68].iloc[i]
    writer.writerow([year,"With incentives","RBC","4.4kWp Battery size = 13.5KWh"])

table=pd.read_csv('period/panel4sb10b135.csv')
for i in range(68):
    print(i)

    year =  table['No solar no incentive MPC'][0:68].iloc[i]
    writer.writerow([year,"Without incentives","MPC","8.8kWp Battery size = 13.5KWh"])

    year =  table['No solar no incentive RBC'][0:68].iloc[i]
    writer.writerow([year,"Without incentives","RBC","8.8kWp Battery size = 13.5KWh"])

    year =  table['No solar with incentive MPC'][0:68].iloc[i]
    writer.writerow([year,"With incentives","MPC","8.8kWp Battery size = 13.5KWh"])

    year =  table['No solar with incentive RBC'][0:68].iloc[i]
    writer.writerow([year,"With incentives","RBC","8.8kWp Battery size = 13.5KWh"])


csvfile.close()

years = pd.read_csv("violin/periodsb10b135.csv")
sns.set(style="whitegrid", palette="pastel", color_codes=True)
#plt.title('Battery size: 13.5 kWh')
ax = sns.catplot(x=" ", y="Payback period",hue="Policy",col="PV system’s size", kind="violin", data=years,palette="BuGn_r",scale="count",cut=0,bw=.2,legend=False)

plt.legend(title="Policy",loc='upper right')
#plt.ylim(12,28)
#plt.title('Battery size: 13.5 kWh',loc='center')
plt.savefig("violin/periodsb10b135.png")
plt.show()
