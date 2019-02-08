import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

#hhids=[86, 59, 77, 26, 93, 101, 114, 171, 1086, 1403]
hhids=['1202', '871', '1103', '585', '59', '2755', '2233', '86', '114', '2575', '1700', '974', '1800',
 '370', '187', '1169', '1718', '545', '94', '2018', '744', '2859', '2925', '484', '2953', '171', '2818', '1953',
 '1697', '1463', '499', '1790', '1507', '1642', '93', '1632',
 '1500', '2472', '2072', '2378', '1415', '2986', '1403', '2945', '77', '1792',
 '624', '379', '2557', '890', '1192', '26', '2787', '2965', '2980', '434', '2829',
 '503', '2532', '946', '2401', '1801','2337','1086','1714','1283','252','2814','101']

scenarios=["sb4b64","sb4b135"]
#compile
for j in scenarios:
    mpc_list=[]
    opt_list=[]
    print("In scenarios",j)
    csvfile = open('hourly_bill/panel4{}.csv'.format(j),'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Home","MPC","Optimal"])
    for i in hhids:

        #MPC
        nj=j[0:2]+'-'+j[2:]
        table = pd.read_csv('mpc_hour/{}_4_mpc_hour/{}.csv'.format(i,nj))
        mpc = table['total_reward'].iloc[-1]
        print("mpc bill: ",mpc)
        mpc_list.append(mpc)


        #optimal bill
        table = pd.read_csv('opt_hour/{}_4_ophour/{}.csv'.format(i,nj))
        opt = table['total_reward'].iloc[8735]-table['total_reward'].iloc[167]
        print("optimal bill: ",opt)
        opt_list.append(opt)
        print("\n")

        writer.writerow([i,mpc,opt])
    writer.writerow(["mean",np.mean(mpc_list),np.mean(opt_list)])
    csvfile.close()
