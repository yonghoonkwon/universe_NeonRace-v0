import argparse
import logging
import sys
import gc
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
from skimage.color import rgb2gray
from skimage.transform import resize

from DQN_breakout.DQN import DQN

ENV_NAME = 'Breakout-v0'  # Environment name
NUM_EPISODES = 12000
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


def main():
    env = gym.make(ENV_NAME)

    print("gi env")
    agent = DQN((FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH), env.action_space.n)

    for _ in range(NUM_EPISODES):
        print('EPISODES ' + str(_))
        t = False
        observation = env.reset()

        state = agent.get_initial_state(observation, observation)
        while not t:
            last_observation = observation
            action = agent.get_action(state)
            observation, reward, terminal, _ = env.step(action)
            # env.render()
            processed_observation = preprocess(observation, last_observation)
            state = np.append(state[1:, :, :], processed_observation, axis=0)


if __name__ == '__main__':
    main()