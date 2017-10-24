import argparse
import logging
import sys
import gc
import cv2
import matplotlib.pyplot as plt
import gym
import universe  # register the universe environments

from universe import wrappers

import numpy as np
import tensorflow as tf
import time
import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.models import load_model

# from model import LSTMPolicy


class Environment(threading.Thread):
    '''
        For running the environment
    '''
    def __init__(self, model, render=True ):
        self.model = model
        self.render = render
        self.stop_signal = False

        # create env


    def runEpisode(self):
        pass


    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True

