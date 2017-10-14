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

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

logger = logging.getLogger()

ENV = 'flashgames.NeonRace-v0'
THREAD_DELAY = 0.001
THREADS = 2
OPTIMIZERS = 1

RUN_TIME = 60
EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

LEARNING_RATE = 0.005
MINI_BATCH_SIZE = 32

GAMMA = 0.99
N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN
NUM_STATE = [396, 622, 3]

left = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
Forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]

ACTIONS = [left, right, Forward]


def crop_screen(screen):

    print(screen.size)
    return screen[84:NUM_STATE[0], 18:NUM_STATE[1], :]



class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', terminal
    lock_queue = threading.Lock()

    def __init__(self, state_size=NUM_STATE, action_space=3):
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

        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1')(l_input)
        x = MaxPool2D((2, 2), strides=(2, 2), name="max_pool_1")(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name="max_pool_2")(x)
        x = Flatten(name='flatten')(x)
        x = Dense(1028, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        out_actions = Dense(self.action_space, activation='softmax')(l_input)
        out_values = Dense(2, activation='linear')(l_input)

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
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MINI_BATCH_SIZE:
                return

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MINI_BATCH_SIZE:
            print("optimizer alert! with bash size %d" % len(s))

        v = self.predict_v(s_)

        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph

        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t:r})

    def train_push(self, s, a, r, s_):
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


class Agent:
    def __init__(self, eps_start, eps_end, eps_step, nb_actions=3):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_step = eps_step
        self.nb_actions = nb_actions
        self.brain = brain

        self.memory = []  # n step return
        self.R = 0.

    def getEpsilon(self):
        if(frames >= eps_step):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps
        # interpolate linearly

    def act(self, s):
        eps = self.getEpsilon()
        global frames;
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, self.nb_actions -1)
        else:
            s = np.array([s])
            p = self.brain.predict_p(s)[0]

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
                self.brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)
            # possible edge case - if an episode ends in <N steps, the computation is incorrect



class Environment(threading.Thread):

    stop_signal = False

    def __init__(self, render=True, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS, nb_actions=3):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(ENV)
        self.env.configure(remotes=1)
        # self.agent = Agent(eps_start, eps_end, eps_steps, brain, nb_actions)

    def runEpisode(self):
        s = self.env.reset()

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            # action_n = [self.env.action_space.sample() for ob in observation_n]
            # print(action_n)
            # observation_n, reward_n, done_n, info = self.env.step(action_n)

            # if (observation_n[0] != None):
            #     print(observation_n[0])
            #     state = observation_n[0]['vision']
            #     # cv2.imshow('image', state)
            #     # plt.imshow(state)
            #     cv2.imwrite('my.jpg', state[84:height, 18:width, :])
            #     break;
            if self.render:
                self.env.render()

            # if s[0] == None:
            #     print("hi")
            #     s, r, done, info = self.env.step([])
            #     continue

            # a = self.agent.act(s)
            # a = 1
            # event_map = ['Forward', 'left', 'right']
            # event = event_map[a]
            # action_n = [event for ob in s]
            # action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in s]
            print(action_n)

            #ACTIONS[a]
            s_, r, done, info = self.env.step([[('KeyEvent', 'ArrowUp', True)] ])
            if (s[0] == None ):
                continue

            # s_ = crop_screen(s_)

            # terminal state
            if done:
                s_ = None

            # self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done:
                print(done, "restart ")
                self.env.reset()
                break

            if self.stop_signal:
                break
            #
            # a = self.agent.act(s)
            # s_, r, done, info = self.env.step(a)
            # s_ = crop_screen(s_)
            #
            # if done:  # terminal state
            #     s_ = None
            #
            # self.agent.train(s, a, r, s_)
            #
            # s = s_
            # R += r
            # if done:
            #     print("dead")


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
            brain.optimize()

    def stop(self):
        self.stop_signal = True




# brain = Brain()  # brain is global in A3C

def main():
    envs = [Environment() for i in range(THREADS)]
    opts = [Optimizer() for i in range(OPTIMIZERS)]

    for e in envs:
        e.start()

    time.sleep(RUN_TIME)

    for e in envs:
        e.stop()
    for e in envs:
        e.join()

        # return 0

if __name__ == '__main__':
    main()
