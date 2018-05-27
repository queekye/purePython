"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import time
from env_euler import Env
from ou_noise import OUNoise
from TerrainMap import TerrainMap
from replay_buffer import ReplayBuffer

#####################  hyper parameters  ####################

MAX_EPISODES = 50000
MAX_EP_STEPS = 200
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.001  # soft replacement
MEMORY_CAPACITY = 10000
REPLAY_START = 1000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'

MAP_DIM = 32
GLOBAL_PIXEL_METER = 4
LOCAL_PIXEL_METER = 1
SIDE = MAP_DIM*GLOBAL_PIXEL_METER
CUT_RATE = MAP_DIM*LOCAL_PIXEL_METER/GLOBAL_PIXEL_METER

###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, m_dim, att_dim):
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound, self.m_dim, self.att_dim = a_dim, s_dim, a_bound, m_dim, att_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.GM = tf.placeholder(tf.float32, [None, m_dim, m_dim, 1], 'r')
        self.LM = tf.placeholder(tf.float32, [None, m_dim, m_dim, 1], 'l')
        self.LM_ = tf.placeholder(tf.float32, [None, m_dim, m_dim, 1], 'l')

        self.a = self._build_a(self.S, self.GM, self.LM, )
        q = self._build_c(self.S, self.GM, self.LM, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, self.GM, self.LM_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, self.GM, self.LM_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s1, gm1, lm1):
        return self.sess.run(self.a, {self.S: s1[np.newaxis, :], self.GM: gm1[np.newaxis, :, :, np.newaxis],
                                      self.LM: lm1[np.newaxis, :, :, np.newaxis]})[0]

    def learn(self):
        replay = self.memory.get_batch(BATCH_SIZE)
        bm_sd = np.asarray([data[0] for data in replay])
        bs = np.asarray([data[1] for data in replay])
        bloc = np.asarray([data[2] for data in replay])
        ba = np.asarray([data[3] for data in replay])
        br = np.asarray([data[4] for data in replay])
        bs_ = np.asarray([data[5] for data in replay])
        bloc_ = np.asarray([data[6] for data in replay])
        bgm = np.zeros([BATCH_SIZE, self.m_dim, self.m_dim, 1])
        blm = np.zeros([BATCH_SIZE, self.m_dim, self.m_dim, 1])
        blm_ = np.zeros([BATCH_SIZE, self.m_dim, self.m_dim, 1])
        for batch in range(BATCH_SIZE):
            sd1 = bm_sd[batch]
            terrian_map = TerrainMap(sd1, MAP_DIM, GLOBAL_PIXEL_METER)
            bgm[batch, :, :, 0] = terrian_map.map_matrix
            blm[batch, :, :, 0] = terrian_map.get_local_map(bloc[batch, :])
            blm_[batch, :, :, 0] = terrian_map.get_local_map(bloc_[batch, :])

        self.sess.run(self.atrain, {self.S: bs, self.GM: bgm, self.LM: blm})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.LM_: blm_})

    def _build_a(self, s, gm, locm, reuse=None, custom_getter=None):

        def _conv2d_keep_size(x, y, kernel_size, name, use_bias=False, reuse_conv=None, trainable_conv=True):
            return tf.layers.conv2d(inputs=x,
                                    filters=y,
                                    kernel_size=kernel_size,
                                    padding="same",
                                    use_bias=use_bias,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    bias_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    reuse=reuse_conv,
                                    name=name,
                                    trainable=trainable_conv)

        def _build_vin(mat, name, trainable_vin):
            h1 = _conv2d_keep_size(mat, 150, 5, name+"_h1", use_bias=True, trainable_conv=trainable_vin)
            r = _conv2d_keep_size(h1, 1, 1, name+"_r", trainable_conv=trainable_vin)
            q0 = _conv2d_keep_size(r, 10, 5, name+"_q0", trainable_conv=trainable_vin)
            v = tf.reduce_max(q0, axis=3, keep_dims=True, name=name+"_v")
            for k in range(30):
                rv = tf.concat([r, v], axis=3)
                q = _conv2d_keep_size(rv, 10, 5, name+"_q", reuse_conv=tf.AUTO_REUSE, trainable_conv=trainable_vin)
                v = tf.reduce_max(q, axis=3, keep_dims=True, name=name+"_v")
            return v

        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            gv = _build_vin(gm, name="global_map_vin", trainable_vin=trainable)
            loc_co = _conv2d_keep_size(locm, 1, 9, name="local_co", use_bias=False, trainable_conv=trainable)
            lv = tf.multiply(gv, loc_co)
            m_flat = tf.reshape(lv, [-1, self.m_dim ** 2])
            att = tf.layers.dense(m_flat, self.att_dim, name='att_l1', trainable=trainable)
            layer_1 = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            layer_2a = tf.layers.dense(layer_1, 600, name='l2a', trainable=trainable)
            layer_2att = tf.layers.dense(att, 600, name='l2att', trainable=trainable)
            layer_2 = tf.add(layer_2a, layer_2att, name="l2")
            layer_3 = tf.layers.dense(layer_2, 600, activation=tf.nn.relu, name='l3', trainable=trainable)
            a1 = tf.layers.dense(layer_3, 4, activation=tf.nn.tanh, name='a1', trainable=trainable)
            a1_norm = tf.nn.l2_normalize(a1, dim=-1)
            a2 = tf.layers.dense(layer_3, 3, activation=tf.nn.tanh, name='a2', trainable=trainable)
            a = tf.concat([a1_norm, a2], axis=-1)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, gm, locm, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            gm_flat = tf.reshape(gm, [-1, self.m_dim**2])
            layer_gm = tf.layers.dense(gm_flat, self.s_dim, activation=tf.nn.relu, name='lgm', trainable=trainable)
            lm_flat = tf.reshape(locm, [-1, self.m_dim ** 2])
            layer_lm = tf.layers.dense(lm_flat, self.s_dim, activation=tf.nn.relu, name='llm', trainable=trainable)
            s_all = tf.concat([layer_gm, layer_lm, s], axis=0)
            layer_1 = tf.layers.dense(s_all, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            layer_2s = tf.layers.dense(layer_1, 600, activation=None, name='l2s', trainable=trainable)
            layer_2a = tf.layers.dense(a, 600, activation=None, name='l2a', trainable=trainable)
            layer_2 = tf.add(layer_2s, layer_2a, name="l2")
            layer_3 = tf.layers.dense(layer_2, 600, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(layer_3, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

env = Env()

s_dim1 = env.s_dim
a_dim1 = env.a_dim
a_bound1 = env.a_bound

ddpg = DDPG(a_dim1, s_dim1, a_bound1, MAP_DIM, att_dim=32)
exploration_noise = OUNoise(a_dim1)  # control exploration
t1 = time.time()
replay_num = 0
env.set_map_seed(187)
for i in range(MAX_EPISODES):
    t_start = time.time()
    sd = i * 3 + 100

    m_sd, s, gm, loc = env.set_state_seed(sd)
    lm = env.get_local_map(loc)
    exploration_noise.reset()
    ep_reward = 0
    ave_dw = 0
    j = 0
    r = 0
    for j in range(MAX_EP_STEPS):
        # Add exploration noise
        a = ddpg.choose_action(s, gm, lm)
        ave_dw += np.linalg.norm(a)
        a += exploration_noise.noise()  # add randomness to action selection for exploration
        a = np.minimum(a_bound1, np.maximum(-a_bound1, a))
        a[0:4] /= max(np.linalg.norm(a[0:4]), 1e-8)

        s_, loc_, r, done = env.step(a)

        ddpg.memory.add(m_sd, s, loc, a, r, s_, loc_)
        replay_num += 1
        if ddpg.pointer > REPLAY_START:
            ddpg.learn()

        s = s_
        loc = loc_
        lm = env.get_local_map(loc_)
        ep_reward += r

        if done:
            break
    ave_dw /= j + 1
    print("episode: %10d   ep_reward:%10.5f   last_reward:%10.5f   replay_num:%10d   "
          "cost_time:%10.2f    ave_dw:" % (i, ep_reward, r, replay_num, time.time() - t_start), ave_dw)

print('Running time: ', time.time() - t1)
