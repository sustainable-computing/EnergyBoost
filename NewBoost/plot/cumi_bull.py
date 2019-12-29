import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

hhids = [59,93,94,101,114,171,187,26,77,86]
#scenarios=["sb4b64","sb4b135","sb8b64","sb8b135","sb10b64","sb10b135","sb20b64","sb20b135"]
scenarios=["sb4b64","sb4b135","sb8b64","sb8b135","sb10b64","sb10b135","sb20b135"]
#hhid = 114
#scenario = "sb8b135"
algos = ["OPTIMAL", "MPC", "RBC", "DLC", "SAC", "A2C", "TD3", "DDPG"]



for hhid in hhids:
    for sc in scenarios:
        fig, ax = plt.subplots()
        #print(str(hhid)+": "+sc)
        for algo in algos:
            table = pd.read_csv("../" + algo + "/result_" + str(hhid) + "/" + sc + ".csv")
            if algo=="OPTIMAL":
                ax.plot(range(8616), (table["Best_Bill"][0:8616]+0.00*np.array(range(8616))), label=algo)
            elif algo=="MPC":
                ax.plot(range(8616), (table["Best_Bill"][0:8616]-0.00*np.array(range(8616))), label=algo)
            else:
                ax.plot(range(8616), (table["Best_Bill"][0:8616]), label=algo)
        ax.set_title(str(hhid)+": "+sc)
        ax.set_ylabel('Cumulative electricity bill ($)',fontsize='15')
        ax.legend(loc=2, borderaxespad=0., bbox_to_anchor=(.00, 1.00), fontsize = '15')
        plt.yticks(fontsize='15')
        plt.xticks(np.arange(0, 8616, 714), ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),rotation=45,fontsize = '15')
        plt.show()

"""
#compile
for j in scenarios:
    print("In scenarios",j)
    csvfile = open('bill_curve/{}.csv'.format(j),'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Home","no_solar","no_battery","baseline","RL","MILP","MPC","optimal"])
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
        nos_curve = table['Base_Bill'][168:8736]-table['Base_Bill'].iloc[167]
        print("no solar bill: ",nos)

        #solar no battery
        if j[2]=='4':
            table = pd.read_csv('nostorage_2/{}_2_nostorage/sb4b0.csv'.format(i))
        if j[2]=='8':
            table = pd.read_csv('nostorage_2/{}_2_nostorage/sb8b0.csv'.format(i))
        if j[2]=='1':
            table = pd.read_csv('nostorage_2/{}_2_nostorage/sb10b0.csv'.format(i))
        if j[2]=='2':
            table = pd.read_csv('nostorage_2/{}_2_nostorage/sb20b0.csv'.format(i))
        nob = table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
        nob_curve = table['Base_Bill'][168:8736]-table['Base_Bill'].iloc[167]
        print("no battery bill: ",nob)


        #Baseline bill
        table = pd.read_csv('rbc_2/{}_2_rbc/{}.csv'.format(i,j))
        bab = table['Base_Bill'].iloc[8735]-table['Base_Bill'].iloc[167]
        bab_curve = table['Base_Bill'][168:8736]-table['Base_Bill'].iloc[167]
        print("Baseline bill: ",bab)

        #RL bill
        table = pd.read_csv('rl_2_bill/{}_2_pre2_bill/{}.csv'.format(i,j))
        beb = table['Bill'].iloc[-1]
        beb_curve = table['Bill'][0:8568]
        print("RL bill: ",beb)

        # #MILP
        # table = pd.read_csv('{}_2_sc_bill/{}.csv'.format(i,j))
        # sc = table['Bill'].iloc[-1]
        # sc_curve = table['Bill'][0:8568]
        # print("MILP bill: ",sc)

        #MPC
        table = pd.read_csv('mpc_2_bill/{}_2_mpc_bill/{}.csv'.format(i,j))
        mpc = table['Bill'].iloc[-1]
        mpc_curve = table['Bill'][0:8568]
        print("mpc bill: ",mpc)

        table = pd.read_csv('olc_2_bill/{}_2_olc_bill/{}.csv'.format(i,j))
        olc = table['Bill'].iloc[-1]
        olc_curve = table['Bill'][0:8568]
        hour = table['Step'][0:8568]
        print("olc bill: ",olc)


        #optimal bill
        nj=j[0:2]+'-'+j[2:]
        table = pd.read_csv('op_2/{}_2_op/{}.csv'.format(i,nj))
        opt2 = table['total_reward'].iloc[8735]-table['total_reward'].iloc[167]


        table = pd.read_csv('op_2_bill/{}_2_op_bill/{}.csv'.format(i,j))
        opt = table['Bill'].iloc[-1]
        opt_curve = table['Bill'][0:8568]
        print("opt bill2",opt2)
        print("opt bill: ",opt)
        # print("optimal bill: ",opt)
        print("\n")
        #writer.writerow([i,nos,nob,bab,beb,sc,mpc,opt])
    csvfile.close()
    fig, ax = plt.subplots()
    #plt.tight_layout()
    #ax.plot(hour, nos_curve, label="No Solar")
    #ax.plot(hour, nob_curve, label="No Battery")
    ax.plot(hour, bab_curve, label="RBC")
    ax.plot(hour, beb_curve, label="RL")
    #ax.plot(hour, sc_curve, label="MILP")
    ax.plot(hour, olc_curve, label="DLC")
    ax.plot(hour, mpc_curve, label="MPC")
    ax.plot(hour, opt_curve, label="Optimal")
    #ax.set_title("Policies, Home Load and PV Output")
    ax.set_ylabel('Cumulative electricity bill ($)',fontsize='15')
    ax.legend(loc=2, borderaxespad=0., bbox_to_anchor=(.00, 1.00), fontsize = '15')
    if j[2]=='4':
        plt.ylim(0,900)
    if j[2]=='8':
        plt.ylim(0,800)
    if j[2]=='1':
        plt.ylim(0,650)
    if j[2]=='2':
        plt.ylim(-200,400)
    #plt.ylim
    plt.yticks(fontsize='15')
    plt.xticks(np.arange(0,8568,714), ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),rotation=45,fontsize = '15')
    plt.savefig('bill_curve/{}_{}_billadded.png'.format(i,j),bbox_inches = "tight")
    #plt.show()
"""
# #plot
# for j in scenarios:
#     d = pd.read_csv('bill/{}.csv'.format(j))
#     no_solar = d['no_solar']
#     #no_battery = d['no_battery']
#     baseline = d['baseline']
#     legend = ['no solar','Baseline', 'SolarReinforce','Optimal']
#     RL = d['RL']
#     optimal = d['optimal']
#     #plt.hist([baseline, RL, optimal], color=['orange', 'green','red'])
#     plt.hist([no_solar, baseline, RL, optimal], color=['navy','orange', 'green','red'])
#     plt.xlabel("Bills")
#     plt.ylabel("Frequency")
#     #plt.axvline(no_solar.mean(), color='sienna', linestyle='dashed', linewidth=1)
#     plt.axvline(no_solar.mean(), color='navy', linestyle='dashed', linewidth=1)
#     plt.axvline(baseline.mean(), color='orange', linestyle='dashed', linewidth=1)
#     plt.axvline(RL.mean(), color='green', linestyle='dashed', linewidth=1)
#     plt.axvline(optimal.mean(), color='red', linestyle='dashed', linewidth=1)
#     plt.legend(legend)
#     #plt.xticks(range(0,7))
#     #plt.yticks(range(1, 20))
#     plt.title('Bills for year 2016\n Scenario: {}'.format(j))
#     #plt.show()
#     plt.savefig('bill/2016_{}_all.png'.format(j))
#     plt.show()
