# !pip install -f https://download.mosek.com/stable/wheel/index.html Mosek
import sys
#sys.path.append("/home/zishan/Documents/")
sys.path.append("/home/azishan/EnergyBoost")

# !pip install cvxopt
#import cvxopt

# !pip install cvxpy
import cvxpy as cvx

#import gurobipy
import copy

import numpy as np
import pandas as pd
import os,sys
import csv
import matplotlib.pyplot as plt
from sklearn.externals import joblib

#input_file = sys.argv[4]
solar_export_rate =  float(sys.argv[1])
b_cap = float(sys.argv[2])
max_rate = float(sys.argv[3])
state=[]
homeid= sys.argv[4].split(".")[2].split("_")[2]
clf_hl = joblib.load('../data/saved_models/hl_rf_{}.pkl'.format(homeid))
clf_ac = joblib.load('../data/saved_models/ghi_rf_{}.pkl'.format(homeid))
print("-homeid-------------------",homeid)


# !pip install scsprox
# !pip install ncvx
# !pip install cylp
# import cylp
# import ncvx

# %matplotlib inline

MAX_TS = 24


# optimizes the policy considering next week's data'
start_point = 0
end_point = 8616
#end_point = 48
#Date:      Jan 1st--------April 30 May 1st---------Oct31 Nov 1st-------Dec 31st
#Time slot:   0----------- 2854     2855----------- 7233 7234----------8745


price_weekday_winter = np.array([0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.132, 0.132, 0.132, 0.132, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.132, 0.132, 0.065, 0.065, 0.065, 0.065, 0.065])
price_weekday_summer = np.array([0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.094, 0.094, 0.094, 0.094, 0.132, 0.132, 0.132, 0.132, 0.132, 0.132, 0.094, 0.094, 0.065, 0.065, 0.065, 0.065, 0.065])
price_weekend = np.tile(np.array([0.065]), 24)

# #winter time
# if (start_point >= 0 and end_point <= 2854) or (start_point >= 7234 and end_point <= 8745):
#     print("winter time")
#     price_weekend = np.tile(np.array([0.065]), 24)
#     price_weekday = np.array([0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.132, 0.132, 0.132, 0.132, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.132, 0.132, 0.065, 0.065, 0.065, 0.065, 0.065])
#     price_week = np.append(np.tile(price_weekday,5), np.tile(price_weekend, 2))
#
# if start_point >= 2855 and end_point <=7233:
#     print("summer time")
#     price_weekend = np.tile(np.array([0.065]), 24)
#     price_weekday = np.array([0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.094, 0.094, 0.094, 0.094, 0.132, 0.132, 0.132, 0.132, 0.132, 0.132, 0.094, 0.094, 0.065, 0.065, 0.065, 0.065, 0.065])
#     price_week = np.append(np.tile(price_weekday,5), np.tile(price_weekend, 2))
#
#
# # weekly price starting from Monday
# P_grid = np.tile(price_week, numweeks)
# #P_grid = np.array([0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.132, 0.132, 0.132, 0.132, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.132, 0.132, 0.065, 0.065, 0.065, 0.065, 0.065])
# #P_table = pd.read_csv('bill/power_price.csv')

P_solar = solar_export_rate*np.ones(MAX_TS)
#P_solar = -P_grid

alpha_c = max_rate # in kW
alpha_d = max_rate # in kW
#b_cap = 6.4 # in kWh
eta_c = 0.95
eta_d = 0.95

E_min = 0
E_max = b_cap
eta_c_leak = 0.0001*b_cap
eta_p_leak = 0

T_u = 1 # in hour
#table = pd.read_csv(input_file)
battery_init = 0.5*b_cap
current_soc = battery_init
total_reward = 0


def cleanup(val):
    if abs(val) < 1e-3:
        return 0
    else:
        return val

def init_ground_truth(datafile):
    print("init_ground_truth")


    if not os.path.exists(datafile):
        print("No datafile was found. Run generatepower.py first.")
        raise ValueError

    with open(datafile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row_count = sum(1 for _ in reader)

    #with open("processed_hhdata_26_result.csv", 'r') as csvfile:
    with open(datafile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row_number = 0

        for row in reader:

            row_number += 1

            if row_number == 1:
                continue
            #print(row)

            state.append(row)


            print("\rEnvironment setup progress: %5.2f%%" % (row_number * 100 / row_count), end='')

    print("\rEnvironment setup finished. Total %i lines data." % row_count)

def predict_day(start):
    day_state=copy.deepcopy(state[start:start+24])
    use_list = np.array([])
    ac_list = np.array([])
    for index in range(24):

        if index==0:
            use = day_state[index][1]
            ac = day_state[index][11]
        else:
            #predict based on previous data
            #localhour	use	temperature	cloud_cover	wind_speed	GH	is_weekday	month	hour	use_hour	use_week	ac	ac_hour	ac_week
            #  0         1        2         3           4       5      6          7      8         9           10       11    12      13
            use = clf_hl.predict([[day_state[index][5], day_state[index-1][1], day_state[index][10], day_state[index][2], day_state[index][3], day_state[index][4], day_state[index][6], day_state[index][7], day_state[index][8]]])[0]
            ac = clf_ac.predict([[day_state[index][1], day_state[index][2], day_state[index][3], day_state[index][4], day_state[index][6], day_state[index-1][11], day_state[index][13], day_state[index][7], day_state[index][8]]])[0]
            #update
            day_state[index][1] = use
            day_state[index][11] = ac
        use_list = np.append(use_list,float(use))
        ac_list = np.append(ac_list,float(ac))

    return use_list, ac_list


def compute(hour_var,battery_var,last):
    #global current_soc
    #global total_reward
    A = np.zeros(MAX_TS)
    L,G = predict_day(hour_var)
    #print("This is L",L)
    #print("This is G",G)
    # L= table['use'][hour_var:MAX_TS+hour_var].values
    # G = table['ac'][hour_var:MAX_TS+hour_var].values
    P_grid = np.array([])
    #P_grid = P_table['price'][hour_var:MAX_TS+hour_var].values
    #for i in range(hour_var,hour_var+24,24):
        #print('first hour of day',i)
        # month = table['month'].iloc[i]
        # hour = table['hour'].iloc[i]
        # week = table['is_weekday'].iloc[i]
    month = float(state[hour_var][7])
    hour = float(state[hour_var][8])
    week = float(state[hour_var][6])


    if month <=4 or month >= 11:
        #print("winter")
        if week:
            #print("weekday")
            P_grid = np.append(P_grid,price_weekday_winter)
        else:
            #print("weekend")
            P_grid = np.append(P_grid,price_weekend)
    else:
        #print("summer")
        if week:
            #print("weekday")
            P_grid = np.append(P_grid,price_weekday_summer)
        else:
            #print("weekend")
            P_grid = np.append(P_grid,price_weekend)



    #print("This is P_grid")
    #print(P_grid)



    E_init = battery_var

    AC = cvx.Variable(MAX_TS,nonneg=True)
    AD = cvx.Variable(MAX_TS,nonpos=True)
    SELECT = cvx.Variable(MAX_TS,boolean=True)

    FG = cvx.Variable(MAX_TS,nonneg=True) # bought from grid
    TG = cvx.Variable(MAX_TS,nonneg=True) # sold to grid
    E = cvx.Variable(MAX_TS+1)

    BS = cvx.Variable(MAX_TS,boolean=True)

    DeltaE = cvx.Variable(MAX_TS+1)
    #Net = cvx.Variable(MAX_TS)

    constraints = []
    constraints += [E[0] == E_init]
    #if last:
    #    constraints += [E[MAX_TS] == battery_init]
    for t in range(MAX_TS):
        constraints += [AD[t] >= -alpha_d*SELECT[t]]
        constraints += [AC[t] <= alpha_c*(1-SELECT[t])]

        constraints += [FG[t]-TG[t] == L[t]+AC[t]+AD[t]-G[t]]
        constraints += [TG[t] <= G[t]*BS[t]]  # maximum power sold back to the grid
        constraints += [FG[t] <= (L[t]+alpha_c)*(1-BS[t])]

        constraints += [DeltaE[t] == T_u*(AC[t]*eta_c + AD[t]/eta_d)]
        constraints += [E[t+1] == E[t]*(1-eta_p_leak)+DeltaE[t]-T_u*eta_c_leak]
        constraints += [E[t+1] <= E_max]
        constraints += [E[t+1] >= E_min]

    # NEM with a different export tariff than ToU
    objective = cvx.Minimize(cvx.sum([FG[t]*P_grid[t]+TG[t]*P_solar[t] for t in range(MAX_TS)]))


    # NEM with the same export tariff as ToU
    # objective = cvx.Minimize(cvx.sum([Net[t]*P_grid[t] for t in range(MAX_TS)]))

    prob = cvx.Problem(objective, constraints)


    # result = prob.solve(solver=cvx.GLPK_MI, verbose=True)
    result = prob.solve(solver=cvx.MOSEK, verbose=False)
    #result = prob.solve(solver=cvx.GUROBI, verbose=True, MIPGap=1e-4)
    #result = prob.solve(solver=cvx.ECOS_BB, verbose=True)

    #print("Bill of first day is",sum([FG[t]*P_grid[t]+TG[t]*P_solar[t] for t in range(24)]).value)
    #print([FG[t].value*P_grid[t]+TG[t].value*P_solar[t] for t in range(24)])
    #first_hour_reward = FG[0].value*P_grid[0]+TG[0].value*P_solar[0]


    #A[0] = cleanup(AC[0].value) + cleanup(AD[0].value)
    for t in range(MAX_TS):
        total_reward += FG[t].value*P_grid[t]+TG[t].value*P_solar[t]
        A[t] = cleanup(AC[t].value) + cleanup(AD[t].value)
        writer.writerow([hour_var+t,A[t],total_reward])
    
    #total_reward = total_reward + first_hour_reward
    current_soc = E[MAX_TS].value


if __name__ == '__main__':
    init_ground_truth(sys.argv[4])
    
    directory="result_{}".format(homeid)
    if not os.path.exists(directory):
        os.makedirs(directory)
    csvfile = open(directory+"/sb".format(homeid)+str(abs(int(float(sys.argv[1])*100)))+"b"+str(int(float(sys.argv[2])*10))+".csv", 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Hour", "Best_Action","Best_Bill"])
    day_count = 1
    for i in range (start_point,end_point,MAX_TS):
        compute(i,current_soc,0)
        print(day_count)
        day_count += 1
    print("total reward is", total_reward)
    csvfile.close()
    #writer.writerow(["total_reward",total_reward])
