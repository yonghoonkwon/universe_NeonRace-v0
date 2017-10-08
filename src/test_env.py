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

env_id = 'flashgames.NeonRace-v0'

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    args = parser.parse_args()


    reg = universe.runtime_spec('flashgames').server_registry
    height = reg[env_id]["height"]
    width = reg[env_id]["width"]


    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)


    env = gym.make(env_id)
    env.configure(remotes=1)  # automatically creates a local docker container

    # Restrict the valid random actions. (Try removing this and see
    # what happens when the agent is given full control of the
    # keyboard/mouse.)
    # env = wrappers.experimental.SafeActionSpace(env)
    observation_n = env.reset()

    while True:
        # your agent here
        #
        # Try sending this instead of a random action: ('KeyEvent', 'ArrowUp', True)
        action_n = [env.action_space.sample() for ob in observation_n]
        print(action_n)
        observation_n, reward_n, done_n, info = env.step(action_n)
        if(observation_n[0] != None):
            print(observation_n[0])
            state = observation_n[0]['vision']
            # cv2.imshow('image', state)
            # plt.imshow(state)
            cv2.imwrite('my.jpg', state[84:height, 18:width, :])
            break;
        print(observation_n)
        env.render()

    # return 0

if __name__ == '__main__':
    main()
    # sys.exit(main())
