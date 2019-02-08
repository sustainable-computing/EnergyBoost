import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
#import tensorflow as tf
import collections
import csv
import os
import pathlib
from environment_bill import EnergyEnvironment

import matplotlib.pyplot as plt
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
  sys.path.append("../")
# from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

#GLOBAL_VARIABLES
MAX_CHARGE_RATE = float(sys.argv[3])
ACTION_BOUND = [-MAX_CHARGE_RATE, MAX_CHARGE_RATE]
#print("0187230981723897",ACTION_BOUND)
current_bill = 0
current_soc = float(sys.argv[2]) * 0.5
# our environment
env = EnergyEnvironment()


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


env.sell_back = float(sys.argv[1])
env.maximum_battery = float(sys.argv[2])
env.battery_starter = env.maximum_battery * 0.5
env.charge_mode = "TOU"
env.datafile = sys.argv[4]
homeid= sys.argv[4].split(".")[0].split("_")[3]
actionlist=pd.read_csv("op_2/{}_2_op/sb-".format(homeid)+str(int(float(sys.argv[1])*100))+"b"+str(int(float(sys.argv[2])*10))+".csv")['Best_Action'][168:]
print("Sell back price is",env.sell_back)
print("battery size is",env.maximum_battery)
env.init_ground_truth()
#env.init_price()
#print out the initial state
#print("inistial state",env.state)
pathlib.Path("op_2_bill/{}_2_op_bill".format(homeid)).mkdir(parents=True, exist_ok=True)
csvfile = open("op_2_bill/{}_2_op_bill/sb".format(homeid)+str(int(float(sys.argv[1])*100))+"b"+str(int(float(sys.argv[2])*10))+".csv", 'w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(["Step", "Action","Bill","reward"])
start_point = 168
end_point = 8736
env.month_starter = start_point
state = env.reset()



bill = compute_bill(env, end_point - start_point)

print("this is best bill",bill)
# sell_back_round=int(float(sys.argv[1])*100)
# battery_round=int(float(sys.argv[2])*10)
