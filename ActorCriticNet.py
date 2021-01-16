import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_HIDDEN_1 = 128
NUM_HIDDEN_2 = 128


class ActorCriticNet:
    # Approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network


    def __init__(self, num_actions, num_states):
        self.num_actions = num_actions
        self.num_states = num_states

    def create_network(self):
        inputs = layers.Input(shape=(self.num_states,))
        dense_1 = layers.Dense(units=NUM_HIDDEN_1, activation='relu')(inputs)
        dense_2 = layers.Dense(units=NUM_HIDDEN_2, activation='relu')(dense_1)
        # The policy and value output
        actor = layers.Dense(units = self.num_actions, activation = 'softmax')(dense_2)
        critic = layers.Dense(units = 1)(dense_2)

        model = keras.Model(inputs, [actor, critic])
        return model