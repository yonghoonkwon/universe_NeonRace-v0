import argparse
import logging
import sys

import cv2
import matplotlib.pyplot as plt
import gym
import universe # register the universe environments

from universe import wrappers

import numpy as np
import tensorflow as tf
import time
import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.models import load_model

logger = logging.getLogger()
# 'flashgames.NeonRace-v0' 'flashgames.YummyyummyMonsterShooter-v0' #'flashgames.3FootNinja-v0'# 'flashgames.CoasterRacer-v0' #
ENV = 'flashgames.NeonRace-v0'  # 'flashgames.SuperCarRacing-v0'  # 'flashgames.V8RacingChampion-v0' # 'flashgames.3dSpeedFever-v0'
THREAD_DELAY = 0.001
THREADS = 4
OPTIMIZERS = 2
LOG_FILE_NAME = str(random.randrange(10,1000))+"_reward.log"

RUN_TIME = 21600
EPS_START = 0.5
EPS_STOP = 0.05
EPS_STEPS = 375000


# EPS_START = 0.000001
# EPS_STOP = 0.0000001
# EPS_STEPS = 75000

LEARNING_RATE = 0.005
MINI_BATCH_SIZE = 32

GAMMA = 0.99
N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN
NUM_STATE = [251, 622, 3]
INPUT_IMG = [84, 206, 6]
NONE_STATE = np.zeros(INPUT_IMG)

left = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
right = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
Forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
ACTIONS = [left, right, Forward]
nb_actions = len(ACTIONS)

frames = 0

def crop_screen(screen):
    # print(screen.shape)
    tmp_img =  screen[229:NUM_STATE[0], 18:NUM_STATE[1], :]

    return cv2.resize(tmp_img, (INPUT_IMG[1], INPUT_IMG[0]), interpolation = cv2.INTER_CUBIC)



class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', terminal
    lock_queue = threading.Lock()

    def __init__(self, state_size=INPUT_IMG, action_space=nb_actions):
        self.session = tf.Session()
        self.state_size = state_size
        self.action_space = action_space

        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

    # init the model of the network
    def _build_model(self):
        l_input = Input(batch_shape=(None, self.state_size[0], self.state_size[1], self.state_size[2]))

        x = Conv2D(16, (4, 4), strides=(4, 4), activation='elu', padding='same', name='conv_1')(l_input)
        x = Dropout(0.5)(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name="max_pool_1")(x)
        x = Conv2D(32, (3, 3), strides=(2, 2), activation='elu', padding='same', name='conv_2')(x)
        print("conv_2", x._keras_shape)
        x = Dropout(0.5)(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name="max_pool_1")(x)
        x = Conv2D(32, (3, 3), strides=(2, 2), activation='elu', padding='same', name='conv_3')(x)
        print("conv_2", x._keras_shape)
        x = Dropout(0.5)(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name="max_pool_3")(x)
        print("conv_3", x._keras_shape)
        x = Flatten(name='flatten')(x)
        
        # print("b4 flattern", x._keras_shape)
        # x = Flatten(name='flatten')(x)
        # print("Flatten(name='flatten')(x)", x._keras_shape)
        # x = Reshape(list(x._keras_shape).insert(0, 0))
        # x = Dropout(0.5)(x)
        # x = LSTM(256, recurrent_dropout=0.0, activation='elu', input_dim =(1,256), name="lstm_1")(x)      
        # x = Reshape(list(x._keras_shape).pop(0))  
        out_actions = Dense(self.action_space, activation='softmax')(x)
        out_values = Dense(1, activation='linear')(x)

        model = Model(inputs=[l_input], outputs=[out_actions, out_values])
        # have to init before threading
        model._make_predict_function()

        return model

    # define the computation graph
    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, self.state_size[0], self.state_size[1], self.state_size[2]))
        a_t = tf.placeholder(tf.float32, shape=(None, self.action_space))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # discounted award

        p, v = model(s_t)  # probi action, values

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantages = r_t - v

        # max policy
        loss_policy = - log_prob * tf.stop_gradient(advantages)

        loss_value = 0.5 * tf.square(advantages)
        entropy = 0.01 * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MINI_BATCH_SIZE :
            # print("len(self.train_queue[0])", len(self.train_queue[0]))
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MINI_BATCH_SIZE:
                # print(" after lock len(self.train_queue[0])", len(self.train_queue[0]))
                return

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]
        # for ms in s:
        #     print(ms.shape)
        s = np.array(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.array(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MINI_BATCH_SIZE:
            print("optimizer alert! with bash size %d" % len(s))

        v = self.predict_v(s_)
        # print(r.shape)
        # print("v", (v).shape)
        # print("GAMMA_N * v", (GAMMA_N * v).shape)
        # print("(GAMMA_N * v * s_mask ).shape", (GAMMA_N * v * s_mask ).shape)

        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph

        # print("-------------- mini loss ------------------------")
        # print(s)
        # print(a)
        # print(r.shape)
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t:r})

    def train_push(self, s, a, r, s_):
        if s == None:
            return
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v

    def save(self):
        self.model.save('my_model.h5')

    def load(self):
        self.model = load_model('my_model.h5')


class Agent:
    def __init__(self, eps_start, eps_end, eps_step, nb_actions=nb_actions):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_step = eps_step
        self.nb_actions = nb_actions

        self.memory = []  # n step return
        self.R = 0.

    def getEpsilon(self):
        if(frames >= self.eps_step):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_step
        # interpolate linearly

    def act(self, s):
        eps = self.getEpsilon()
        global frames, brain;
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, self.nb_actions -1)
        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]
            # print(self.nb_actions, p)
            a = np.random.choice(self.nb_actions, p=p)
            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        # turn action in to one-hot representation
        a_cats =  np.zeros(self.nb_actions)
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N)/ GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)
            # possible edge case - if an episode ends in <N steps, the computation is incorrect



class Environment(threading.Thread):

    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS, nb_actions=nb_actions):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(ENV)
        self.env.configure(remotes=1)
        self.agent = Agent(eps_start, eps_end, eps_steps, nb_actions)

    def runEpisode(self):
        s = self.env.reset()[0]

        R = 0
        start = time.time()
        while True:
            time.sleep(THREAD_DELAY)  # yield

            if self.render:
                self.env.render()

            if (s != None):
                a = self.agent.act(s)
            else:
                a = 2
            # action_n = [ for ob in s]
            # print(action_n)
            s_, r, done, info = self.env.step([ACTIONS[a]])
            s_ = s_[0]
            done = done[0]
            if (s_ == None):
                start = time.time()
                continue

            if(time.time() -start > 160):
                done = True

            s_ = crop_screen(s_['vision'])
            # cv2.imwrite('my.jpg', s_)
            # print(done)


            # terminal state
            if done:
                s_ = None
                print(done, "restart ")
                self.env.reset()
                break

            if self.stop_signal:
                print("stop signal")
                break

            self.agent.train(s, a, r[0], np.dstack([s, s_]))
            # print(r, done)

            s = s_
            R += r[0]

        with open(LOG_FILE_NAME,'a') as f:
            f.write(str(R)+"\n")
        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            # print("------------------------opt------------------------")
            brain.optimize()

    def stop(self):
        self.stop_signal = True




brain = Brain()  # brain is global in A3C
# brain.load()

def main():
    envs = [Environment() for i in range(THREADS)]
    opts = [Optimizer() for i in range(OPTIMIZERS)]

    for o in opts:
        o.start()

    for e in envs:
        e.start()

    time.sleep(RUN_TIME)

    for e in envs:
        e.stop()
    for e in envs:
        e.join()

    for o in opts:
        o.stop()
    for o in opts:
        o.join()

    brain.save()

if __name__ == '__main__':
    main()
