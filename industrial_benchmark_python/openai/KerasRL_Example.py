import os
import sys

# FIXMERuntime environment dependent paths ...
this_files_dir = os.path.dirname(os.path.realpath(__file__))
# We just assume we are in the siemens/industrialbenchmark sources
project_source_root = os.path.join(this_files_dir, '../..')

sys.path.append("../..")
sys.path.append("..")
sys.path.append("../goldstone")
sys.path.append(project_source_root)

import IndustrialBenchmarkEnv

# Create the environment
sim_props_file = os.path.join(project_source_root, 'src/main/resources/sim.properties')
# NOTE This is not supported yet! WE can only use the default values as of now
#   (same as in src/main/ressources/simTest.properties)
sim_props_file = os.path.join(project_source_root, 'src/main/resources/sim.properties')
# HACK therefore:
sim_props_file = None
env = IndustrialBenchmarkEnv.IndustrialBenchmarkEnv(None)

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env2 = gym.make(ENV_NAME)
print('action space', env2.action_space)
print('observation space', env2.observation_space.shape[0])
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = 4

# Option 1 : Simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + (4,)))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

# Option 2: deep network
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('softmax'))


print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
cem.fit(env, nb_steps=100000, visualize=False, verbose=2)

# After training is done, we save the best weights.
cem.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=5, visualize=True)
