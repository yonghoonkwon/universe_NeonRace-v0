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
RUN_TIME = 60
EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000


class Environment(threading.Thread):

    stop_signal = False

    def __init__(self, render=True, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(ENV)
        self.env.configure(remotes=1)

    def runEpisode(self):
        observation_n = self.env.reset()

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            action_n = [self.env.action_space.sample() for ob in observation_n]
            print(action_n)
            observation_n, reward_n, done_n, info = self.env.step(action_n)

            # if (observation_n[0] != None):
            #     print(observation_n[0])
            #     state = observation_n[0]['vision']
            #     # cv2.imshow('image', state)
            #     # plt.imshow(state)
            #     cv2.imwrite('my.jpg', state[84:height, 18:width, :])
            #     break;

            if self.render:
                self.env.render()

            if done_n:
                print(done_n, "restart ")
                self.env.reset()

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


def main():
    envs = [Environment() for i in range(THREADS)]

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
