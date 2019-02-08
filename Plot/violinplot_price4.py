import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
csvfile = open("violin/violin_policy4.csv", 'w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(["Total electricity bill of one year ($)","Panel number","Battery size","Policy"])
#hhids=['1202', '871', '1103', '585', '59', '2755', '2233', '86', '114', '2575', '1700', '974', '1800','2557', '890', '1192', '26', '2787', '2965', '2980', '434', '2829','503', '2532', '946', '2401', '1801','101','2337','1086','1714','1283','252','2814']
hhids=['1202', '871', '1103', '585', '59', '2755', '2233', '86', '114', '2575', '1700', '974', '1800',
 '370', '187', '1169', '1718', '545', '94', '2018', '744', '2859', '2925', '484', '2953', '171', '2818', '1953',
 '1697', '1463', '499', '1790', '1507', '1642', '93', '1632',
 '1500', '2472', '2072', '2378', '1415', '2986', '1403', '2945', '77', '1792',
 '624', '379', '2557', '890', '1192', '26', '2787', '2965', '2980', '434', '2829',
 '503', '2532', '946', '2401', '1801','2337','1086','1714','1283','252','2814']
for i in hhids:
    print(i)
    table=pd.read_csv("nopv_4/{}_4_nopv/sb4b0.csv".format(i))
    bill=table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
    # writer.writerow([bill/1.3,0,0,"RBC"])
    # writer.writerow([bill/1.3,0,6.4,"RBC"])
    # writer.writerow([bill/1.3,0,13.5,"RBC"])
    # writer.writerow([bill/1.3,0,0,"RL"])
    # writer.writerow([bill/1.3,0,6.4,"RL"])
    # writer.writerow([bill/1.3,0,13.5,"RL"])
    # writer.writerow([bill/1.3,0,0,"MPC"])
    # writer.writerow([bill/1.3,0,6.4,"MPC"])
    # writer.writerow([bill/1.3,0,13.5,"MPC"])
    # writer.writerow([bill/1.3,0,0,"Optimal"])
    # writer.writerow([bill/1.3,0,6.4,"Optimal"])
    # writer.writerow([bill/1.3,0,13.5,"Optimal"])

    table = pd.read_csv('nostorage_2/{}_2_nostorage/sb4b0.csv'.format(i))
    bill = table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
    # writer.writerow([bill/1.3,2,0,"RBC"])
    # writer.writerow([bill/1.3,2,0,"RL"])
    # writer.writerow([bill/1.3,2,0,"Optimal"])
    # writer.writerow([bill/1.3,2,0,"MPC"])

    table = pd.read_csv('rbc_2/{}_2_rbc/sb4b64.csv'.format(i))
    bill = table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
    writer.writerow([bill/1.3,2,6.4,"RBC"])
    table = pd.read_csv('rbc_2/{}_2_rbc/sb4b135.csv'.format(i))
    bill = table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
    #writer.writerow([bill/1.3,2,13.5,"RBC"])


    table = pd.read_csv('rl_2_bill/{}_2_pre2_bill/sb4b64.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    writer.writerow([bill/1.3,2,6.4,"RL"])
    table = pd.read_csv('rl_2_bill/{}_2_pre2_bill/sb4b135.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    #writer.writerow([bill/1.3,2,13.5,"RL"])


    table = pd.read_csv('olc_2_bill/{}_2_olc_bill/sb4b64.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    writer.writerow([bill/1.3,2,6.4,"DLC"])
    table = pd.read_csv('olc_2_bill/{}_2_olc_bill/sb4b135.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    #writer.writerow([bill/1.3,2,13.5,"MPC"])



    table = pd.read_csv('mpc_2_bill/{}_2_mpc_bill/sb4b64.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    writer.writerow([bill/1.3,2,6.4,"MPC"])
    table = pd.read_csv('mpc_2_bill/{}_2_mpc_bill/sb4b135.csv'.format(i))
    bill = table['Bill'].iloc[-1]
    #writer.writerow([bill/1.3,2,13.5,"MPC"])


    table = pd.read_csv('op_2/{}_2_op/sb-4b64.csv'.format(i))
    bill = table['total_reward'].iloc[8735]-table['total_reward'].iloc[167]
    writer.writerow([bill/1.3,2,6.4,"Optimal"])
    table = pd.read_csv('op_2/{}_2_op/sb-4b135.csv'.format(i))
    bill = table['total_reward'].iloc[8735]-table['total_reward'].iloc[167]
    #writer.writerow([bill/1.3,2,13.5,"Optimal"])




csvfile.close()


bills = pd.read_csv("violin/violin_policy4.csv")
sns.set(style="whitegrid", palette="pastel", color_codes=True)
#ax = sns.violinplot(x="sell_back_price", y="bill",hue="panel_number", data=bills,palette="muted")
#ax = sns.catplot(x="sell_back_price", y="bill",hue="panel_number", kind="swarm", data=bills,palette="muted");
#ax = sns.catplot(x="Panel number", y="Bill", col="Method",hue="Battery size", kind="violin", data=bills,palette="muted");
ax = sns.catplot(x="Policy", y="Total electricity bill of one year ($)", kind="violin", data=bills,palette="Blues",scale="count",cut=0,bw=.2);
plt.ylim(-750,1750)



plt.savefig("violin/4-controller_added.png")
plt.show()
