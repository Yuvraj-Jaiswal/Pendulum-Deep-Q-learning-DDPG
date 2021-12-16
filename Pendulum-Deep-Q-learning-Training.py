import gym
import random

env = gym.make("Pendulum-v0")
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]

##

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential,Model

actionInput = Input(shape=actions)
observation_input = Input(shape=(1,states))
flattened_observation = Flatten()(observation_input)

actor = Sequential()
actor.add(Flatten(input_shape=(1,states)))
actor.add(Dense(24,activation="relu"))
actor.add(Dense(24,activation="relu"))
actor.add(Dense(24,activation="relu"))
actor.add(Dense(actions,activation="linear"))

x = Concatenate()([actionInput,flattened_observation])
x = Dense(32,activation='relu')(x)
x = Dense(32,activation='relu')(x)
x = Dense(32,activation='relu')(x)
x = Dense(1,activation='linear')(x)
critic = Model(inputs=[actionInput, observation_input], outputs=x)

memory = SequentialMemory(limit=50000, window_length=1)
agent = DDPGAgent(nb_actions=actions , actor=actor , critic=critic ,critic_action_input=actionInput ,
                  memory=memory)

agent.compile(Adam(lr=1e-3),metrics=["mae"])
agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=200)

##
import numpy as np
Test_agent = agent.test(env,nb_episodes=10)
print(np.mean(Test_agent.history['episode_reward']))

##
agent.save_weights("AgentWeight")