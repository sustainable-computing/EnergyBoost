import sys
#sys.path.append("/home/zishan/Documents/")
sys.path.append("/home/azishan/EnergyBoost/")


import itertools
import matplotlib
import numpy as np
import pandas as pd
import torch
import collections
import csv
import os

from NewBoost.env.environment import EnergyEnvironment
from NewBoost.lib import plotting
from ourDDPG import DDPG as Agent

import utils

import matplotlib.pyplot as plt
import sklearn.pipeline
import sklearn.preprocessing

#if "../" not in sys.path:
#    sys.path.append("../")
# from lib.envs.cliff_walking import CliffWalkingEnv
#from lib import plotting


from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

#GLOBAL_VARIABLES
BATCH_SIZE = 256
MAX_CHARGE_RATE = float(sys.argv[3])
ACTION_BOUND = [-MAX_CHARGE_RATE, MAX_CHARGE_RATE]
#print("0187230981723897",ACTION_BOUND)
current_bill = 0
current_soc = float(sys.argv[2]) * 0.5

# our environment
env = EnergyEnvironment(MAX_CHARGE_RATE)

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = pd.read_csv(sys.argv[4])[0:8640][['use','ac','hour','month','is_weekday']]
#observation_examples = np.array([env.state[5]])
observation_examples = observation_examples.values
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
     ("rbf1", RBFSampler(gamma=5.0, n_components=200)),
     ("rbf2", RBFSampler(gamma=2.0, n_components=200)),
     ("rbf3", RBFSampler(gamma=1.0, n_components=200)),
     ("rbf4", RBFSampler(gamma=0.5, n_components=200))
     ])
featurizer.fit(scaler.transform(observation_examples))


def featurize_state(state):
    """
    RBF feature representation of a given state

    :param state: current state

    :return: state with new feature representation
    """
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]

def run(env, month_var, battery_var, agent, num_episodes, discount_factor=1.0):

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    env.month_starter = month_var
    env.battery_starter = battery_var
    global current_bill
    best_reward = 0
    best_battery = 0
    best_actions = []
    best_bill = []
    env.ulist=[]
    env.alist=[]
    #print("======the updated action bound is========098q20938109", ACTION_BOUND)
    #agent.__init__(800,1,2)
    replay_buffer = utils.ReplayBuffer(800, 1)
    total_steps = 0
    #state = env.reset()
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        # print("month_starter",env.month_starter)
        # print("battery_starter",env.battery_starter)
        # print("current battery",env.state[env.current_index][8])
        # print("current state",env.state[env.current_index])
        # print("MAX_CHARGE_RATE",MAX_CHARGE_RATE)
        # print("======the updated action bound is========", ACTION_BOUND)
        #print("\n=================================")
        state = env.reset()
        
        #agent.reset()
        #print("The state is\n",env.state)
        #print("current index",env.current_index)
        ep_reward = 0
        actions = []
        total_bill = []

        #print("\n")
        # print(state)
        # print("---------------------")

        episode = []

        # One step in the environment
        for t in itertools.count():
            # print("In episode:",i_episode)
            # print("The step",t)
            #print("state current_index",env.current_index)
            # print("The state is ",env.state)
            # env.render()

            ACTION_BOUND = [-min(env.state[env.current_index][8], env.state[env.current_index][5], MAX_CHARGE_RATE), min((env.maximum_battery - env.state[env.current_index][8]), MAX_CHARGE_RATE)]
            #ACTION_BOUND = [-MAX_CHARGE_RATE, MAX_CHARGE_RATE]
            # estimator_policy.lower = ACTION_BOUND[0]
            # estimator_policy.higher = ACTION_BOUND[1]
            # if t==0:
            #     #print("==========================================")
            #     print("month_starter",env.month_starter)
            #     print("battery_starter",env.battery_starter)
            #     print("current battery",env.state[env.current_index][8])
            #     print("current state",env.state[env.current_index])
            #     print("======the updated action bound is========", ACTION_BOUND)



            # Take a step
            #action_learn = estimator_policy.predict(state)
            #state = world.get_state()
            state = featurize_state(state)
            
#            if total_steps < 1000:
#                action_learn = env.action_space.sample()
#            else:
#                action_learn = agent.select_action(state)
                
            action_learn = (
                agent.select_action(state)
                + np.random.normal(0, MAX_CHARGE_RATE * 0.1, size=1)
            ).clip(-MAX_CHARGE_RATE, MAX_CHARGE_RATE)
            
            action = np.clip(action_learn,*ACTION_BOUND)
            
            
            #agent.train(replay_buffer, BATCH_SIZE)
            
            _, next_state, reward, done = env.step(action)

            #mask = 1 if i_episode == num_episodes else float(not done)

            replay_buffer.add(state, action, featurize_state(next_state), reward, done)
            
            if total_steps > BATCH_SIZE:
                agent.train(replay_buffer, BATCH_SIZE)

            total_steps += 1
            
            
            #agent.step(state, action, reward, featurize_state(next_state), done)
            #agent.train()
            #action_learn = agent.noise_action(state)
            # print("policy learned action",action_learn)
            #action = np.clip(action_learn,*ACTION_BOUND)
            # print("real action",action)

            #print("this is action",action)
            actions.append(action[0])
            #tng, next_state, reward, done = env.step(action)
            #agent.perceive(state,action,reward,next_state,done)
            # print("tng is", tng)
            # print("next_state is", next_state)
            # print("reward is", reward)
            # print("it is done or not", done)

            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            ep_reward +=reward
            stats.episode_lengths[i_episode] = t

#            # Calculate TD Target
#            value_next = estimator_value.predict(next_state)
#            td_target = reward + discount_factor * value_next
#            td_error = td_target - estimator_value.predict(state)

#            # Update the value estimator
#            estimator_value.update(state, td_target)

#            # Update the policy estimator
#            # using the td error as our advantage estimate
#            estimator_policy.update(state, td_error, action)

            # Print out which step we're on, useful for debugging.
            #print("\rStep {} @ Episode {}/{} ({})".format(\
            #        t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))

            state = next_state
            total_bill.append(-ep_reward)
            if done or t==MAX_EP_STEPS-1:
                break

        if i_episode == 0 or ep_reward > best_reward:
            best_actions = actions[:]
            best_reward = ep_reward
            best_bill = total_bill[:]
            best_battery = env.state[env.current_index][8]

        if i_episode == num_episodes - 1:
            for i in range(len(best_actions)):
                # print("this is index---------",i)
                # print("action",best_actions[i])
                # print("bill",best_bill[i])
                writer.writerow([month_var+i,best_actions[i],current_bill+best_bill[i]])

            current_bill = current_bill+(-best_reward)
            current_soc = best_battery



    return stats

if __name__ == '__main__':

    #MAX_EP_STEPS = 24*7*4*3 #season
    #MAX_EP_STEPS = 24*7 #week
    MAX_EP_STEPS = 24 #Day
    #MAX_EP_STEPS = 1 #Hour
    env.sell_back = float(sys.argv[1])
    env.maximum_battery = float(sys.argv[2])
    env.battery_starter = env.maximum_battery * 0.5
    env.charge_mode = "TOU"
    env.datafile = sys.argv[4]
    homeid= sys.argv[4].split(".")[2].split("_")[2]
    print("Sell back price is",env.sell_back)
    print("battery size is",env.maximum_battery)
    env.init_ground_truth()
    #env.init_price()
    #print out the initial state
    #print("inistial state",env.state)
    directory="result_{}".format(homeid)
    if not os.path.exists(directory):
        os.makedirs(directory)
    #csvfile = open(directory+"/sb".format(homeid)+str(int(float(sys.argv[1])*100))+"b"+str(int(float(sys.argv[2])*10))+".csv", 'w', newline='')
    csvfile = open(directory+"{}_{}_{}_{}".format(str(int(float(sys.argv[1])*100)), str(int(float(sys.argv[2])*10)), str(int(sys.argv[5])), str(int(sys.argv[6])))+".csv", 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Hour", "Best_Action", "Best_Bill"])
    start_point = int(sys.argv[5])
    end_point = int(sys.argv[6])

    
    #print('Testing')
    #print(env.reset())
    #sys.exit()
    #with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        # Note, due to randomness in the policy the number of episodes you need varies
        # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
    state_dim = len(featurize_state(env.reset()))
    #test = np.array([-2,2])
    #print(test.high)
    #sys.exit()
    
    #args = utils.Args()
    #agent = Agent(state_dim=state_dim, action_dim=1, max_action=MAX_CHARGE_RATE*1.05)
    for i in range (start_point,end_point,MAX_EP_STEPS):
        #sess.run(tf.initialize_all_variables())
        agent = Agent(state_dim=state_dim, action_dim=1, max_action=MAX_CHARGE_RATE)
        stats = run(env, i, current_soc, agent, 500, discount_factor=1)

    print("This is the best bill",current_bill)
    sell_back_round=int(float(sys.argv[1])*100)
    battery_round=int(float(sys.argv[2])*10)
    # plot the stat results
    plotting.plot_episode_stats(stats,homeid,sell_back_round,battery_round,name=directory+'/DDPG',smoothing_window=10)
