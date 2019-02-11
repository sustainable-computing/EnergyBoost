# !pip install -f https://download.mosek.com/stable/wheel/index.html Mosek
#import mosek

# !pip install cvxopt
import cvxopt

# !pip install cvxpy
import cvxpy as cvx

import gurobipy

import numpy as np
import pandas as pd
import os,sys
import csv
import matplotlib.pyplot as plt

input_file = sys.argv[4]
solar_export_rate =  float(sys.argv[1])
b_cap = float(sys.argv[2])
max_rate = float(sys.argv[3])
homeid= sys.argv[4].split(".")[0].split("_")[3]
print("-homeid-------------------",homeid)


# !pip install scsprox
# !pip install ncvx
# !pip install cylp
# import cylp
# import ncvx

# %matplotlib inline

def cleanup(val):
    if abs(val) < 1e-3:
        return 0
    else:
        return val

numweeks = 1
numdays = 7 * numweeks
MAX_TS = 24 * numdays


# optimizes the policy considering next week's data'
start_point = 0
#end_point = 24*7*51
end_point = 24*7*52
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
table = pd.read_csv(input_file)
battery_init = 0.5*b_cap
current_soc = battery_init
total_reward= 0


def compute(hour_var,battery_var,last):
    global current_soc
    global total_reward
    A = np.zeros(MAX_TS)
    twouse = table['use'][hour_var:MAX_TS+hour_var]
    L = twouse*2
    L = L.values
    G = table['AC'][hour_var:MAX_TS+hour_var].values
    P_grid = np.array([])
    #P_grid = P_table['price'][hour_var:MAX_TS+hour_var].values
    for i in range(hour_var,hour_var+24*7*1,24):
        #print('first hour of day',i)
        month = table['month'].iloc[i]
        hour = table['hour'].iloc[i]
        week = table['is_weekday'].iloc[i]

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



    # E_init = battery_var
    #
    # AC = cvx.Variable(MAX_TS,nonneg=True)
    # AD = cvx.Variable(MAX_TS,nonpos=True)
    # SELECT = cvx.Variable(MAX_TS,boolean=True)
    #
    # FG = cvx.Variable(MAX_TS,nonneg=True) # bought from grid
    # TG = cvx.Variable(MAX_TS,nonneg=True) # sold to grid
    # E = cvx.Variable(MAX_TS+1)
    #
    # BS = cvx.Variable(MAX_TS,boolean=True)
    #
    # DeltaE = cvx.Variable(MAX_TS+1)
    # #Net = cvx.Variable(MAX_TS)
    #
    # constraints = []
    # constraints += [E[0] == E_init]
    # if last:
    #     constraints += [E[MAX_TS] == battery_init]
    # for t in range(MAX_TS):
    #     constraints += [AD[t] >= -alpha_d*SELECT[t]]
    #     constraints += [AC[t] <= alpha_c*(1-SELECT[t])]
    #
    #     constraints += [FG[t]-TG[t] == L[t]+AC[t]+AD[t]-G[t]]
    #     constraints += [TG[t] <= G[t]*BS[t]]  # maximum power sold back to the grid
    #     constraints += [FG[t] <= (L[t]+alpha_c)*(1-BS[t])]
    #
    #     constraints += [DeltaE[t] == T_u*(AC[t]*eta_c + AD[t]/eta_d)]
    #     constraints += [E[t+1] == E[t]*(1-eta_p_leak)+DeltaE[t]-T_u*eta_c_leak]
    #     constraints += [E[t+1] <= E_max]
    #     constraints += [E[t+1] >= E_min]
    #
    # # NEM with a different export tariff than ToU
    # objective = cvx.Minimize(cvx.sum([FG[t]*P_grid[t]+TG[t]*P_solar[t] for t in range(MAX_TS)]))
    #
    #
    # # NEM with the same export tariff as ToU
    # # objective = cvx.Minimize(cvx.sum([Net[t]*P_grid[t] for t in range(MAX_TS)]))
    #
    # prob = cvx.Problem(objective, constraints)
    #
    #
    # # result = prob.solve(solver=cvx.GLPK_MI, verbose=True)
    # # result = prob.solve(solver=cvx.MOSEK, verbose=True)
    # result = prob.solve(solver=cvx.GUROBI, verbose=True, MIPGap=1e-4)
    # # result = prob.solve(solver=cvx.ECOS_BB, verbose=True)
    #
    # print("Bill of first day is",sum([FG[t]*P_grid[t]+TG[t]*P_solar[t] for t in range(24)]).value)
    # print([FG[t].value*P_grid[t]+TG[t].value*P_solar[t] for t in range(24)])
    #
    #
    #
    #
    # total_reward = total_reward + objective.value
    # current_soc = E[MAX_TS-1].value

    for t in range(MAX_TS):
        #print(["Time slot", hour_var+t+1, "charge power is", cleanup(AC[t].value),"discharge power is", cleanup(AD[t].value)])
        writer.writerow([hour_var+t+1,P_grid[t]])


if __name__ == '__main__':
    # directory="{}_4_sc".format(homeid)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    csvfile = open("price_tou.csv", 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Hour", "price"])# FG stands for
    for i in range (start_point,end_point,MAX_TS):
        #print("this is index",i)
        if i==end_point-MAX_TS:
            compute(i,current_soc,1)
        else:
            compute(i,current_soc,0)
    print("total reward is", total_reward)
    csvfile.close()
    #writer.writerow(["total_reward",total_reward])
