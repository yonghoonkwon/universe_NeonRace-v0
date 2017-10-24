import argparse
import logging
import sys
import gc
import cv2
import matplotlib.pyplot as plt
import gym
import universe # register the universe environments

from universe import wrappers
from collections import deque

from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import time
import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.models import load_model


LEARNING_RATE = 0.005
MOMENTUM = 0.2
MIN_GRAD = 0.0001
ENV_NAME = 'break_out'
SAVE_SUMMARY_PATH = './logs'
SAVE_NETWORK_PATH = './network'
LOAD_NETWOROK = False
INITIAL_REPLAY_SIZE = 200000 # Nb steps for memory, before training
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
TRAIN_INTERVAL = 1000
GAMMA = 0.99  # Discount factor
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
FRAME_WIDTH = 84
FRAME_HEIGHT = 84

class DQN:

    def __init__(self, input_shape, nb_actions,
                 init_epsilon=1.0,
                 final_epsilon=0.1,
                 exploration_steps=1000000):

        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.final_epsilon = final_epsilon

        self.epsilon = init_epsilon
        self.epsilon_step = (init_epsilon - final_epsilon) / exploration_steps
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # create replay memory
        self.replay_memory = deque()

        # create network
        self.state, self.q_vals, self.network = self._build_network()
        q_network_weights = self.network.trainable_weights

        # create target network
        self.state_t, self.q_vals_t, self.network_t = self._build_network()
        q_network_weights_t = self.network_t.trainable_weights

        # define copy operation
        self.update_target_network = [q_network_weights_t[i].assign(q_network_weights[i]) for i in range(len(q_network_weights_t))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self._build_train_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self._build_summary()
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.global_variables_initializer())

        if LOAD_NETWOROK:
            self._load_netowrk()

        self.sess.run(self.update_target_network)

    def _build_network(self):
        model = Sequential()
        model.add(Conv2D(32, 8, strides=(4, 4), activation='relu', input_shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2]]))
        model.add(Conv2D(64, 4, strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, 3, strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.nb_actions))

        state = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        q_vals = model(state)

        return state, q_vals, model

    def _build_train_op(self, network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # convert into to one hot
        a_one_hot = tf.one_hot(a, self.nb_actions, 1.0, 0.)
        q_value = tf.reduce_sum(tf.multiply(self.q_vals, a_one_hot), reduction_indices=1)

        # clip the error
        error = tf.abs(y - q_value)
        clipped = tf.clip_by_value(error, 0.0, 1.0)
        linear = error - clipped
        loss = tf.reduce_mean(0.5 * tf.square(clipped) + linear)

        rms_optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = rms_optimizer.minimize(loss, var_list=network_weights)

        return a, y, loss, grads_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def _build_summary(self):

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]

        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_test(self, state):
        return np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.nb_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > self.final_epsilon and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def _train(self):
        s_batch = []
        a_batch = []
        r_batch = []
        s__batch = []
        t_batch = []
        y_batch = []

        # sample from memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            s_batch.append(data[0])
            a_batch.append(data[1])
            r_batch.append(data[2])
            s__batch.append(data[3])
            t_batch.append(data[4])

        # bool to int
        t_batch = np.array(t_batch) + 0

        next_actions_batch = np.argmax(self.q_vals.eval(feed_dict={self.s: s__batch}), axis=1)
        target_q_values_batch = self.q_vals_t.eval(feed_dict={self.s_t: s__batch})
        for i in range(len(minibatch)):
            y_batch.append(r_batch[i] + (1 - t_batch[i]) * GAMMA * target_q_values_batch[i][next_actions_batch[i]])

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(s_batch) / 255.0),
            self.a: a_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def add_memory(self, s, a, r, t, s_):
        next_state = np.append(s[1:, :, :], s_, axis=0)

        # clip reward into -1,1
        reward = np.clip(r, -1, 1)

        # add into replay memory
        self.replay_memory.append((s, a, next_state, t))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY :
            self.replay_memory.popleft()

        if self.t > INITIAL_REPLAY_SIZE:
            # train network
            if self.t % TRAIN_INTERVAL == 0:
                self._train()

            # update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # save network
            if self.t % SAVE_INTERVAL == 0:
                s_path = self.saver.save(self.sess, SAVE_NETWORK_PATH, global_step=self.t)
                print('saved network')

        self.total_reward += reward
        self.total_q_max += np.max(self.q_vals.eval(feed_dict={self.s: [np.float32(s / 255.0)]}))
        self.duration += 1

        if t:
            # write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max/float(self.duration),
                         self.duration, self.total_loss/ (float(self.duration)/ float(TRAIN_INTERVAL))]

                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})

                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1
        return next_state