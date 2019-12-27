import sys
sys.path.append("/home/zishan/Documents/")
#sys.path.append("/home/azishan/")

#import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
#import tensorflow as tf
import collections
import csv
import os
import glob
import pathlib

from NewBoost.env.environment import EnergyEnvironment

import matplotlib.pyplot as plt
import sklearn.pipeline
import sklearn.preprocessing

#if "../" not in sys.path:
#  sys.path.append("../")
# from lib.envs.cliff_walking import CliffWalkingEnv
# from lib import plotting

from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')


def compute_bill(env, length):
    current_bill=0
    #state = env.reset()
    #print(state)
    for t in range(length):
        #print(t)
        ACTION_BOUND = [-min(env.state[env.current_index][8], env.state[env.current_index][5], MAX_CHARGE_RATE), min((env.maximum_battery - env.state[env.current_index][8]), MAX_CHARGE_RATE)]

        action = actionlist.iloc[t]
        action = np.clip(action,*ACTION_BOUND)
        tng, next_state, reward, done = env.step(action)

        #update bill
        current_bill += (-reward)
        #print("\rStep {} reward ({})".format(t, reward, end=""))
        writer.writerow([t, action, current_bill,-reward])

    return current_bill


hhids = [59]
for i in hhids:
    directory = "result_{}".format(i)
    file_list = glob.glob(directory+"/*.csv")
    for j in file_list:
        #GLOBAL_VARIABLES
        MAX_CHARGE_RATE = 2 if int(j.split('b')[2].split('.')[0])==64 else 5
        ACTION_BOUND = [-MAX_CHARGE_RATE, MAX_CHARGE_RATE]
        
        current_bill = 0
        current_soc = (float(j.split('b')[2].split('.')[0])/10.0) * 0.5
        # our environment
        env = EnergyEnvironment(MAX_CHARGE_RATE)

        env.sell_back = float(j.split('b')[1])/100.0
        env.maximum_battery = float(j.split('b')[2].split('.')[0])/10.0
        env.battery_starter = env.maximum_battery * 0.5
        env.charge_mode = "TOU"
        env.datafile = "../data/added_hhdata_"+str(i)+"_2.csv"
        homeid = i
        actionlist=pd.read_csv(j)['Best_Action'][0:8616]
        print("Sell back price is",env.sell_back)
        print("battery size is",env.maximum_battery)
        env.init_ground_truth()
        #env.init_price()
        #print out the initial state
        #print("inistial state",env.state)
        directory = "converted_{}/".format(i)
        if not os.path.exists(directory):
            os.makedirs(directory)
        csvfile = open(directory+j.split("/")[1], 'w', newline='')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Step", "Action","Best_Bill","reward"])
        start_point = 0
        end_point = 8616
        env.month_starter = start_point
        state = env.reset()



        bill = compute_bill(env, end_point - start_point)

        print("this is best bill",bill)
        # sell_back_round=int(float(sys.argv[1])*100)
        # battery_round=int(float(sys.argv[2])*10)
    
