import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

hhids=[26, 59, 77, 86, 93, 94,101]
scenarios=["sb4b64","sb4b135","sb8b64","sb8b135","sb10b64","sb10b135","sb20b64","sb20b135"]
csvfile = open('bill_curve/nem_results.csv','w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(["scenarios","Optimal","MPC","Difference"])
for j in scenarios:
    nj=j[0:2]+'-'+j[2:]
    op_list=[]
    mpc_list=[]
    print("In scenarios",j)
    for i in hhids:
        table = pd.read_csv('op_nem_2/{}_2_opnem/{}.csv'.format(i,nj))
        opt = table['total_reward'].iloc[8735]-table['total_reward'].iloc[167]
        op_list.append(opt)

        table = pd.read_csv('mpc_nem_2/{}_2_mpc_nem/{}.csv'.format(i,nj))
        mpc = table['total_reward'].iloc[-1]
        mpc_list.append(mpc)

    writer.writerow([j,np.mean(op_list),np.mean(mpc_list),np.mean(op_list) - np.mean(mpc_list)])

csvfile.close()
