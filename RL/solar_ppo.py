import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import collections
#from collections import namedtuple
import csv
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
from environment import EnergyEnvironment

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

EP_MAX = 1000
EP_LEN = 24 #day
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 5, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


# our environment
env = EnergyEnvironment()

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
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    #print("this featurized",featurized)
    #print("this is featurized[0]",featurized[0])
    return featurized[0]



class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        #print("action inside choose is",a)
        return np.clip(a, *ACTION_BOUND) # check if bouned action works

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

def run_ppo(env, month_var, battery_var):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a critic
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(EP_MAX),
        episode_rewards=np.zeros(EP_MAX))

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

    for ep in range(EP_MAX):
        # Reset the environment and pick the fisrst action
        # print("month_starter",env.month_starter)
        # print("battery_starter",env.battery_starter)
        # print("current battery",env.state[env.current_index][8])
        # print("current state",env.state[env.current_index])
        # print("MAX_CHARGE_RATE",MAX_CHARGE_RATE)
        # print("======the updated action bound is========", ACTION_BOUND)
        print("\n=================================")
        state = env.reset()
        #print("The state is\n",env.state)
        # print("current index",env.current_index)
        actions = []
        total_bill = []

        # print("\n")
        # print(state)
        # print("---------------------")

        episode = []
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0

        # One step in the environment
        for t in range(EP_LEN): # in one episode
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
            action = ppo.choose_action(state)
            # print("policy learned action",action_learn)
            # action = np.clip(action_learn,*ACTION_BOUND)
            # print("real action",action)

            #actions.append(action[0])
            tng, next_state, reward, done = env.step(action)
            # print("tng is", tng)
            # print("next_state is", next_state)
            # print("reward is", reward)
            # print("it is done or not", done)


            buffer_s.append(state)
            buffer_a.append(action)
            buffer_r.append(reward)    # normalize reward, find to be useful
            state = next_state
            ep_r += reward

            # # Keep track of the transition
            # episode.append(Transition(
            #   state=state, action=action, reward=reward, next_state=next_state, done=done))
            #
            # # Update statistics
            # stats.episode_rewards[i_episode] += reward
            # ep_reward +=reward
            # stats.episode_lengths[i_episode] = t

            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN-1:
                v_s_ = ppo.get_v(next_state)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)
        if ep == 0: all_ep_r.append(ep_r)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )


        # if i_episode == 0 or ep_reward > best_reward:
        #     best_actions = actions[:]
        #     best_reward = ep_reward
        #     best_bill = total_bill[:]
        #     best_battery = env.state[env.current_index][8]
        #
        # if i_episode == num_episodes - 1:
        #     for i in range(len(best_actions)):
        #         # print("this is index---------",i)
        #         # print("action",best_actions[i])
        #         # print("bill",best_bill[i])
        #         writer.writerow([month_var+i,best_actions[i],current_bill+best_bill[i]])
        #
        #     current_bill = current_bill+(-best_reward)
        #     current_soc = best_battery

        if ep == EP_MAX - 1:
            # for i in range(len(best_actions)):
            #     # print("this is index---------",i)
            #     # print("action",best_actions[i])
            #     # print("bill",best_bill[i])
            #     writer.writerow([month_var+i,actions[i],current_bill+total_bill[i]])

            current_bill = current_bill+(-ep_r)
            current_soc = env.state[env.current_index][8]

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()

    return stats


# tf.reset_default_graph()
#
# global_step = tf.Variable(0, name="global_step", trainable=False)
# policy_estimator = PolicyEstimator(learning_rate=0.001)
# value_estimator = ValueEstimator(learning_rate=0.1)
ppo = PPO()
all_ep_r = []
#MAX_EP_STEPS = 24*7*4*3 #season
#MAX_EP_STEPS = 24*7 #week
MAX_EP_STEPS = 24 #Day
#MAX_EP_STEPS = 1 #Hour
env.sell_back = float(sys.argv[1])
env.maximum_battery = float(sys.argv[2])
env.battery_starter = env.maximum_battery * 0.5
env.charge_mode = "TOU"
env.datafile = sys.argv[4]
homeid= sys.argv[4].split(".")[0].split("_")[3]
print("Sell back price is",env.sell_back)
print("battery size is",env.maximum_battery)
env.init_ground_truth()
#env.init_price()
#print out the initial state
#print("inistial state",env.state)
directory="{}_2_pre2TEST".format(homeid)
if not os.path.exists(directory):
    os.makedirs(directory)
csvfile = open("{}_2_pre2TEST/sb".format(homeid)+str(int(float(sys.argv[1])*100))+"b"+str(int(float(sys.argv[2])*10))+".csv", 'w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(["Hour", "Best_Action", "Best_Bill"])
start_point = 72
end_point = 72+24
#end_point = 8616

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Note, due to randomness in the policy the number of episodes you need varies
    # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
    for i in range (start_point,end_point,MAX_EP_STEPS):
        #sess.run(tf.initialize_all_variables())
        stats = run_ppo(env, i, current_soc)

print("this is best bill",current_bill)
csvfile.close()
sell_back_round=int(float(sys.argv[1])*100)
battery_round=int(float(sys.argv[2])*10)
# plot the stat results
#plotting.plot_episode_stats(stats,homeid,sell_back_round,battery_round,smoothing_window=10)
