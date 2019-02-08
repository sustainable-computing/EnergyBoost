#import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
import csv
import shutil
import math
import pathlib
from environment_base import EnergyEnvironment

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

# # all placeholder for tf
# with tf.name_scope('S'):
#     S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
# with tf.name_scope('R'):
#     R = tf.placeholder(tf.float32, [None, 1], name='r')
# with tf.name_scope('S_'):
#     S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')

# class Actor(object):
#     def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
#         self.sess = sess
#         self.a_dim = action_dim
#         self.action_bound = action_bound
#         self.lr = learning_rate
#         self.t_replace_iter = t_replace_iter
#         self.t_replace_counter = 0
#
#         with tf.variable_scope('Actor'):
#             # input s, output a
#             self.a = self._build_net(S, scope='eval_net', trainable=True)
#
#             # input s_, output a, get a_ for critic
#             self.a_ = self._build_net(S_, scope='target_net', trainable=False)
#
#         self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
#         self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
#
#     def _build_net(self, s, scope, trainable):
#         with tf.variable_scope(scope):
#             init_w = tf.contrib.layers.xavier_initializer()
#             init_b = tf.constant_initializer(0.001)
#             net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
#                                   kernel_initializer=init_w, bias_initializer=init_b, name='l1',
#                                   trainable=trainable)
#             net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
#                                   kernel_initializer=init_w, bias_initializer=init_b, name='l2',
#                                   trainable=trainable)
#             net = tf.layers.dense(net, 10, activation=tf.nn.relu,
#                                   kernel_initializer=init_w, bias_initializer=init_b, name='l3',
#                                   trainable=trainable)
#             with tf.variable_scope('a'):
#                 actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
#                                           name='a', trainable=trainable)
#                 scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
#         return scaled_a
#
#     def learn(self, s):   # batch update
#         self.sess.run(self.train_op, feed_dict={S: s})
#         if self.t_replace_counter % self.t_replace_iter == 0:
#             self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
#         self.t_replace_counter += 1
#
#     def choose_action(self, s):
#         s = s[np.newaxis, :]    # single state
#         return self.sess.run(self.a, feed_dict={S: s})[0]  # single action
#
#     def add_grad_to_graph(self, a_grads):
#         with tf.variable_scope('policy_grads'):
#             self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
#
#         with tf.variable_scope('A_train'):
#             opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
#             self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))
#
#
# class Critic(object):
#     def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
#         self.sess = sess
#         self.s_dim = state_dim
#         self.a_dim = action_dim
#         self.lr = learning_rate
#         self.gamma = gamma
#         self.t_replace_iter = t_replace_iter
#         self.t_replace_counter = 0
#
#         with tf.variable_scope('Critic'):
#             # Input (s, a), output q
#             self.a = a
#             self.q = self._build_net(S, self.a, 'eval_net', trainable=True)
#
#             # Input (s_, a_), output q_ for q_target
#             self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net
#
#             self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
#             self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')
#
#         with tf.variable_scope('target_q'):
#             self.target_q = R + self.gamma * self.q_
#
#         with tf.variable_scope('TD_error'):
#             self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))
#
#         with tf.variable_scope('C_train'):
#             self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
#
#         with tf.variable_scope('a_grad'):
#             self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)
#
#     def _build_net(self, s, a, scope, trainable):
#         with tf.variable_scope(scope):
#             init_w = tf.contrib.layers.xavier_initializer()
#             init_b = tf.constant_initializer(0.01)
#
#             with tf.variable_scope('l1'):
#                 n_l1 = 200
#                 w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
#                 w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
#                 b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
#                 net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
#             net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
#                                   kernel_initializer=init_w, bias_initializer=init_b, name='l2',
#                                   trainable=trainable)
#             net = tf.layers.dense(net, 10, activation=tf.nn.relu,
#                                   kernel_initializer=init_w, bias_initializer=init_b, name='l3',
#                                   trainable=trainable)
#             with tf.variable_scope('q'):
#                 q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
#         return q
#
#     def learn(self, s, a, r, s_):
#         self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
#         if self.t_replace_counter % self.t_replace_iter == 0:
#             self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
#         self.t_replace_counter += 1
#
#
# class Memory(object):
#     def __init__(self, capacity, dims):
#         self.capacity = capacity
#         self.data = np.zeros((capacity, dims))
#         self.pointer = 0
#
#     def store_transition(self, s, a, r, s_):
#         transition = np.hstack((s, a, [r], s_))
#         index = self.pointer % self.capacity  # replace the old memory with new memory
#         self.data[index, :] = transition
#         self.pointer += 1
#
#     def sample(self, n):
#         assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
#         indices = np.random.choice(self.capacity, size=n)
#         return self.data[indices, :]


# sess = tf.Session()
#
# # Create actor and critic.
# actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
# critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
# actor.add_grad_to_graph(critic.a_grads)
#
# M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)
#
# saver = tf.train.Saver()
# path = '.\Mode'
#
# if LOAD:
#     saver.restore(sess, tf.train.latest_checkpoint(path))
# else:
#     sess.run(tf.global_variables_initializer())


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

            if s[7] ==0: #weekends
                a=0
            else: #weekdays
                if s[9]>=5 and s[9]<=10: #summer
                    if s[0]>=11 and s[0] < 17:
                        if s[8] >= MAX_CHARGE_RATE:
                            a = -MAX_CHARGE_RATE
                        else:
                            a = -s[8]
                    if s[0] < 7 or s[0] >=19:
                        if env.maximum_battery - s[8] >= MAX_CHARGE_RATE:
                            a = MAX_CHARGE_RATE
                        else:
                            a = env.maximum_battery - s[8]


                else: #winter
                    if (s[0]>=7 and s[0] < 11) or (s[0]>=17 and s[0] < 19):
                        if s[8] >= MAX_CHARGE_RATE:
                            a = -MAX_CHARGE_RATE
                        else:
                            a = -s[8]
                    if s[0] < 7 or s[0] >=19:
                        if MAX_CHARGE_RATE !=0:
                            k_star=math.ceil((env.maximum_battery-s[8])/(MAX_CHARGE_RATE*0.95))
                            if 7-k_star == s[0]:
                                a = (((env.maximum_battery-s[8])-((k_star-1)*MAX_CHARGE_RATE*0.95))/0.95)
                            if s[0] > 7-k_star and s[0] < 7:
                                a = MAX_CHARGE_RATE
                            else:
                                a=0
                        else:
                            a=0



            if ep==0:
                actions_base.append(a)
            # 1 if it is ddpg, 0 for not
            tng,s_, r, done = env.step(a)
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


# def eval():
#     s = env.reset()
#     done = False
#     ep_reward = 0
#     all_actions = []
#     for ep in range(MAX_EP_STEPS):
#         a = actor.choose_action(s)
#         all_actions.append(a[0])
#         s_, r, done = env.step(a)
#         s = s_
#         ep_reward += r
#     print(ep_reward)
#     print(all_actions)



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
        # directory="{}_2_base".format(homeid)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        pathlib.Path("base_2/{}_2_base".format(homeid)).mkdir(parents=True, exist_ok=True)
        csvfile = open("base_2/{}_2_base/sb".format(homeid)+str(int(float(sys.argv[1])*100))+"b"+str(int(float(sys.argv[2])*10))+".csv", 'w', newline='')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Hour", "Base_Action","Base_Bill","Base_Grid","Base_Soc"])
        pre_train(1)
        csvfile.close()
