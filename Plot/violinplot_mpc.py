import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
csvfile = open("violin/violin_mpc_sb4.csv", 'w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(["Total electricity bill of one year ($)","PV system’s size (kWp)","Battery size","Method"])
hhids=['1202', '871', '1103', '585', '59', '2755', '2233', '86', '114', '2575', '1700', '974', '1800','2557', '890', '1192', '26', '2787', '2965', '2980', '434', '2829','503', '2532', '946', '2401', '1801','101','2337','1086','1714','1283','252','2814']
hhids=['1202', '871', '1103', '585', '59', '2755', '2233', '86', '114', '2575', '1700', '974', '1800',
 '370', '187', '1169', '1718', '545', '94', '2018', '744', '2859', '2925', '484', '2953', '171', '2818', '1953',
 '1697', '1463', '499', '1790', '1507', '1642', '93', '1632',
 '1500', '2472', '2072', '2378', '1415', '2986', '1403', '2945', '77', '1792',
 '624', '379', '2557', '890', '1192', '26', '2787', '2965', '2980', '434', '2829',
 '503', '2532', '946', '2401', '1801','2337','1086','1714','1283','252','2814','101']
for i in hhids:
    print(i)

    table = pd.read_csv('nostorage_2/{}_2_nostorage/sb4b0.csv'.format(i))
    bill = table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
    writer.writerow([bill/1.3,4.4,"no battery","MPC"])
    table = pd.read_csv('nostorage_4/{}_4_nostorage/sb4b0.csv'.format(i))
    bill = table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
    writer.writerow([bill/1.3,8.8,"no battery","MPC"])


    table = pd.read_csv('mpc_2_bill/{}_2_mpc_bill/sb4b64.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    writer.writerow([bill/1.3,4.4,"6.4 kWh","MPC"])
    table = pd.read_csv('mpc_2_bill/{}_2_mpc_bill/sb4b135.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    writer.writerow([bill/1.3,4.4,"13.5 kWh","MPC"])

    table = pd.read_csv('mpc_4_bill/{}_4_mpc_bill/sb4b64.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    writer.writerow([bill/1.3,8.8,"6.4 kWh","MPC"])
    table = pd.read_csv('mpc_4_bill/{}_4_mpc_bill/sb4b135.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    writer.writerow([bill/1.3,8.8,"13.5 kWh","MPC"])






csvfile.close()


bills = pd.read_csv("violin/violin_mpc_sb4.csv")
sns.set(style="whitegrid", palette="pastel", color_codes=True)
#ax = sns.violinplot(x="sell_back_price", y="bill",hue="panel_number", data=bills,palette="muted")
#ax = sns.catplot(x="sell_back_price", y="bill",hue="panel_number", kind="swarm", data=bills,palette="muted");
#ax = sns.catplot(x="Panel number", y="Bill", col="Method",hue="Battery size", kind="violin", data=bills,palette="muted");
#ax = sns.catplot(x="Battery size", y="Bill",col="Panel size", kind="violin", data=bills,palette="muted");
ax = sns.catplot(x="PV system’s size (kWp)", y="Total electricity bill of one year ($)",hue="Battery size", kind="violin", data=bills,palette="BuGn_r",scale="count",cut=0,bw=.2,legend=False);
plt.legend(title="Battery size",loc='upper right')

plt.savefig("violin/sb4-newvio.png")
plt.show()
