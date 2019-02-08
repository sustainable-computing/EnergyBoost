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
 '503', '2532', '946', '2401', '1801','2337','1086','1714','1283','252','2814']

scenarios=["sb4b64","sb4b135","sb8b64","sb8b135","sb10b64","sb10b135","sb20b64","sb20b135"]
#compile
for j in scenarios:
    nos_list=[]
    nob_list=[]
    nob_fg_list=[]
    nob_tg_list=[]
    rbc_list=[]
    rbc_fg_list=[]
    rbc_tg_list=[]
    mpc_list=[]
    mpc_fg_list=[]
    mpc_tg_list=[]

    print("In scenarios",j)
    csvfile = open('roi_table_all/panel4{}.csv'.format(j),'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Home","no_solar","no_battery","NoB FG","NoB TG","RBC","RBC FG","RBC TG","MPC","MPC FG","MPC TG"])
    for i in hhids:
        #no solar no battery
        #table = pd.read_csv('nopv_4/{}_2_nopv/{}.csv'.format(i,j))
        if j[2]=='4':
            table = pd.read_csv('nopv_4/{}_4_nopv/sb4b0.csv'.format(i))
        if j[2]=='8':
            table = pd.read_csv('nopv_4/{}_4_nopv/sb8b0.csv'.format(i))
        if j[2]=='1':
            table = pd.read_csv('nopv_4/{}_4_nopv/sb10b0.csv'.format(i))
        if j[2]=='2':
            table = pd.read_csv('nopv_4/{}_4_nopv/sb20b0.csv'.format(i))
        nos = table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
        print("no solar bill: ",nos)
        nos_list.append(nos)

        #solar no battery
        if j[2]=='4':
            table = pd.read_csv('nostorage_4/{}_4_nostorage/sb4b0.csv'.format(i))
        if j[2]=='8':
            table = pd.read_csv('nostorage_4/{}_4_nostorage/sb8b0.csv'.format(i))
        if j[2]=='1':
            table = pd.read_csv('nostorage_4/{}_4_nostorage/sb10b0.csv'.format(i))
        if j[2]=='2':
            table = pd.read_csv('nostorage_4/{}_4_nostorage/sb20b0.csv'.format(i))
        nob = table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
        print("no battery bill: ",nob)
        nob_list.append(nob)
        nob_fg = nos
        nob_fg_list.append(nob_fg)
        nob_tg = -1*(nob_fg-nob)
        nob_tg_list.append(nob_tg)


        #Baseline bill
        table = pd.read_csv('rbc_4_bill/{}_4_rbc_bill/{}.csv'.format(i,j))
        rbc = table['Bill'].iloc[-1]
        rbc_fg = table['Total FG bill'].iloc[-1]
        rbc_tg = table['Total TG bill'].iloc[-1]
        print(" bill: ",rbc)
        rbc_list.append(rbc)
        rbc_fg_list.append(rbc_fg)
        rbc_tg_list.append(rbc_tg)



        #MPC
        table = pd.read_csv('mpc_4_par/{}_4_mpc_par/{}.csv'.format(i,j))
        mpc = table['Bill'].iloc[-1]
        mpc_fg = table['Total FG bill'].iloc[-1]
        mpc_tg = table['Total TG bill'].iloc[-1]
        print("mpc bill: ",mpc)
        mpc_list.append(mpc)
        mpc_fg_list.append(mpc_fg)
        mpc_tg_list.append(mpc_tg)
        print("\n")

        writer.writerow([i,nos,nob,nob_fg,nob_tg,rbc,rbc_fg,rbc_tg,mpc,mpc_fg,mpc_tg])
    writer.writerow(["mean",np.mean(nos_list),np.mean(nob_list),np.mean(nob_fg_list),np.mean(nob_tg_list),np.mean(rbc_list),np.mean(rbc_fg_list),np.mean(rbc_tg_list),np.mean(mpc_list),np.mean(mpc_fg_list),np.mean(mpc_tg_list)])
    csvfile.close()
