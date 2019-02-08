import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
import csv
import shutil
import math
from environment_nostorage import EnergyEnvironment

MAX_EPISODES = 1
MAX_EP_STEPS = 24 * 30
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 1.0  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 168
BATCH_SIZE = 5000
VAR_MIN = 0.02
LOAD = False
TEMP = []
actions_base = []
bill_base = []
grid_base = []
soc_base = []
current_soc=float(sys.argv[2])/2
current_bill=0
current_soc=0
#policy = pd.read_csv('86_4/sb20b135.csv')['Best_Action']


env = EnergyEnvironment()

STATE_DIM = 10
ACTION_DIM = 1
MAX_CHARGE_RATE = float(sys.argv[3])
ACTION_BOUND = [-MAX_CHARGE_RATE, MAX_CHARGE_RATE]

def pre_train(episodes):
    global actions_base
    global bill_base
    global grid_base
    global soc_base
    actions_base=[]
    bill_base=[]
    soc_base=[]
    for ep in range(episodes):
        grid_base=[]
        s = env.reset()
        ep_reward = 0

        for t in range(MAX_EP_STEPS):
            a=0

            if ep==0:
                actions_base.append(a)
            # 1 if it is ddpg, 0 for not
            tng,s_, r, done = env.step()
            #print("this is base grid", tng)
            grid_base.append(tng)
            soc_base.append(s_[8])

            s = s_
            ep_reward += r
            if ep==0:
                bill_base.append(-ep_reward)

            if t == MAX_EP_STEPS-1 or done:
                result = '| done' if done else '| ----'
                print('Pre-Ep:', ep,
                      result,
                      '| R: %.4f' % ep_reward)
                break

    for i in range(len(actions_base)):
        writer.writerow([i,actions_base[i],bill_base[i],grid_base[i],soc_base[i]])
    print("len of best actions",len(actions_base))
    print("len of best bill",len(bill_base))
    print("Generating baseline done.")


def eval():
    s = env.reset()
    done = False
    ep_reward = 0
    all_actions = []
    for ep in range(MAX_EP_STEPS):
        a = actor.choose_action(s)
        all_actions.append(a[0])
        s_, r, done = env.step(a)
        s = s_
        ep_reward += r
    print(ep_reward)
    print(all_actions)



if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        #train(2, [0.33717483][0], 0)
        MAX_EP_STEPS = 24*7*4*13
        #env.two_meter = False
        env.sell_back = float(sys.argv[1])
        env.maximum_battery = float(sys.argv[2])
        env.charge_mode = "TOU"
        env.datafile = sys.argv[4]
        homeid= sys.argv[4].split(".")[0].split("_")[4]
        print("Running on home ----------------------------------------",homeid)
        print("Sell back price is",env.sell_back)
        print("battery size is",env.maximum_battery)
        env.init_ground_truth()
        # initial_size=float(sys.argv[2])/2
        # print("initial size is",initial_size)
        directory="{}_2_nostorage".format(homeid)
        if not os.path.exists(directory):
            os.makedirs(directory)
        csvfile = open("{}_2_nostorage/sb".format(homeid)+str(int(float(sys.argv[1])*100))+"b"+str(int(float(sys.argv[2])*10))+".csv", 'w', newline='')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Hour", "Base_Action","Base_Bill","Base_Grid","Base_Soc"])
        pre_train(1)
        csvfile.close()
