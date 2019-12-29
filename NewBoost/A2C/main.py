import sys
#sys.path.append("/home/zishan/Documents/")
sys.path.append("/home/azishan/EnergyBoost/")
 
#import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import collections
import csv
import os

from NewBoost.env.environment import EnergyEnvironment
from NewBoost.lib import plotting
# from environment import EnergyEnvironment
#
import matplotlib.pyplot as plt
import sklearn.pipeline
import sklearn.preprocessing
#
# if "../" not in sys.path:
#   sys.path.append("../")
# # from lib.envs.cliff_walking import CliffWalkingEnv
# from lib import plotting
#
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

#GLOBAL_VARIABLES
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

class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [800], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.mu = tf.squeeze(self.mu)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.lower = ACTION_BOUND[0]
            self.higher = ACTION_BOUND[1]
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist._sample_n(1)
            #print("----------------read here------------------")
            self.action = tf.clip_by_value(self.action, self.lower, self.higher)

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.action) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        """
        predict the action given a state
        :param state: current state
        :param sess: tensorflow session
        :return: Updated session
        """
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.action, { self.state: state })

    def update(self, state, target, action, sess=None):
        """
        Update the policy function
        :param state: current state
        :param target: td target
        :param action: learned action
        :param sess: tensorflow session
        :return: loss
        """
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [800], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        """
        predict the value given a state
        :param state: current state
        :param sess: tensorflow session
        :return: Updated session
        """
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        """
        Update the policy function
        :param state: current state
        :param target: td target
        :param action: learned action
        :param sess: tensorflow session
        :return: loss
        """
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def actor_critic(env, month_var, battery_var, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.
    :param env: OpenAI environment.
    :param month_var: index of month
    :param battery_var: current battery SoC
    :param estimator_policy: Policy Function to be optimized
    :param estimator_value: Value function approximator, used as a critic
    :param num_episodes: Number of episodes to run for
    :param discount_factor: Time-discount factor
    :return: An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # #pretrain
    # ACTION_BOUND = [-MAX_CHARGE_RATE,MAX_CHARGE_RATE]
    # for i_episode in range(10):
    #
    #     state = env.reset()
    #     ep_reward = 0
    #
    #     # One step in the environment
    #     for t in range(len(env.state)-1):
    #
    #         ACTION_BOUND = [-min(env.state[env.current_index][8], MAX_CHARGE_RATE), min((env.maximum_battery - env.state[env.current_index][8]), MAX_CHARGE_RATE)]
    #
    #
    #         # Take a step
    #         action_learn = estimator_policy.predict(state)
    #         action = np.clip(action_learn,*ACTION_BOUND)
    #
    #
    #
    #         #print("this is action",action)
    #         tng, next_state, reward, done = env.step(action)
    #
    #         #update stats
    #         ep_reward +=reward
    #
    #         # Calculate TD Target
    #         value_next = estimator_value.predict(next_state)
    #         td_target = reward + discount_factor * value_next
    #         td_error = td_target - estimator_value.predict(state)
    #
    #         # Update the value estimator
    #         estimator_value.update(state, td_target)
    #
    #         # Update the policy estimator
    #         # using the td error as our advantage estimate
    #         estimator_policy.update(state, td_error, action)
    #
    #         # Print out which step we're on, useful for debugging.
    #         print("\rStep {} @ Episode {}/{}".format(t, i_episode + 1, 10))
    #
    #         if t== (len(env.state)-2):
    #             break
    #
    #         state = next_state
    # print("pretrain finished")

    # Keeps track of useful statistics
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
            action_learn = estimator_policy.predict(state)
            # print("policy learned action",action_learn)
            action = np.clip(action_learn,*ACTION_BOUND)
            # print("real action",action)

            #print("this is action",action)
            actions.append(action[0])
            tng, next_state, reward, done = env.step(action)
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

            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)

            # Update the value estimator
            estimator_value.update(state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(state, td_error, action)

            # Print out which step we're on, useful for debugging.
            #print("\rStep {} @ Episode {}/{} ({})".format(
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

    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator(learning_rate=0.001)
    value_estimator = ValueEstimator(learning_rate=0.1)
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
    csvfile = open(directory+"/sb".format(homeid)+str(int(float(sys.argv[1])*100))+"b"+str(int(float(sys.argv[2])*10))+".csv", 'w', newline='')
    #csvfile = open(directory+"{}_{}_{}_{}".format(str(int(float(sys.argv[1])*100)), str(int(float(sys.argv[2])*10)), str(int(sys.argv[5])), str(int(sys.argv[6])))+".csv", 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Hour", "Best_Action", "Best_Bill"])
    #start_point = int(sys.argv[5])
    #end_point = int(sys.argv[6])
    start_point = 0
    end_point = 8616

    day_count = 1
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        # Note, due to randomness in the policy the number of episodes you need varies
        # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
        for i in range (start_point,end_point,MAX_EP_STEPS):
            sess.run(tf.initialize_all_variables())
            stats = actor_critic(env, i, current_soc, policy_estimator, value_estimator, 100, discount_factor=1)
            print(day_count)
            day_count += 1

    print("this is best bill",current_bill)
    sell_back_round=int(float(sys.argv[1])*100)
    battery_round=int(float(sys.argv[2])*10)
    # plot the stat results
    plotting.plot_episode_stats(stats,homeid,sell_back_round,battery_round,name=directory+'/A2C_'+str(homeid)+"_"+str(sell_back_round)+"_"+str(battery_round), smoothing_window=10)
