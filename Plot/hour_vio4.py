import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

#hhids=[86, 59, 77, 26, 93, 101, 114, 171, 1086, 1403]
hhids=['1202', '871', '1103', '585', '59', '2755', '2233', '86', '114', '2575', '1700', '974', '1800',
 '370', '187', '1169', '1718', '545', '94', '2018', '744', '2859', '2925', '484', '2953', '171', '2818', '1953',
 '1697', '1463', '499', '1790', '1507', '1642', '93', '1632',
 '1500', '2472', '2072', '2378', '1415', '2986', '1403', '2945', '77', '1792',
 '624', '379', '2557', '890', '1192', '26', '2787', '2965', '2980', '434', '2829',
 '503', '2532', '946', '2401', '1801','2337','1086','1714','1283','252','2814','101']


#compile
csvfile = open('violin/panel4.csv','w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(["Total electricity bill of one year ($)","Method","Battery size"])

for i in hhids:

    #MPC
    table = pd.read_csv('mpc_hour/{}_4_mpc_hour/sb-4b64.csv'.format(i))
    mpc = table['total_reward'].iloc[-1]
    print("mpc bill: ",mpc)
    writer.writerow([mpc,"MPC","6.4 kWh"])

    table = pd.read_csv('mpc_hour/{}_4_mpc_hour/sb-4b135.csv'.format(i))
    mpc = table['total_reward'].iloc[-1]
    print("mpc bill: ",mpc)
    writer.writerow([mpc,"MPC","13.5 kWh"])


    #optimal bill
    table = pd.read_csv('opt_hour/{}_4_ophour/sb-4b64.csv'.format(i))
    opt = table['total_reward'].iloc[8735]-table['total_reward'].iloc[167]
    print("optimal bill: ",opt)
    writer.writerow([opt,"Optimal","6.4 kWh"])

    table = pd.read_csv('opt_hour/{}_4_ophour/sb-4b135.csv'.format(i))
    opt = table['total_reward'].iloc[8735]-table['total_reward'].iloc[167]
    print("optimal bill: ",opt)
    writer.writerow([opt,"Optimal","13.5 kWh"])
    print("\n")

#writer.writerow(["mean",np.mean(mpc_list),np.mean(opt_list)])
csvfile.close()

bills = pd.read_csv("violin/panel4.csv")
sns.set(style="whitegrid", palette="pastel", color_codes=True)
ax = sns.catplot(x="Battery size", y="Total electricity bill of one year ($)",hue="Method", kind="violin", data=bills,palette="Blues",scale="count",cut=0,bw=.2,legend=False)
plt.legend(title="Method",loc='upper right')
plt.savefig("violin/panel4.png")
plt.show()
