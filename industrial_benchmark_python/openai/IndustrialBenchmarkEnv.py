# coding=utf-8
'''
The MIT License (MIT)

Copyright 2017 Siemens AG

Authors: Robin Vobruba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import jpype
import gym
import os
import sys
import random
import gym
from gym import spaces
from IDS import IDS
from NamedShape import NamedShape
import numpy as np
import time
import collections

# Input vars
DELTA_VELOCITY = "DeltaVelocity"
DELTA_GAIN = "DeltaGain"
DELTA_SHIFT = "DeltaShift"

# Output vars
ACTION_VELOCITY = "Velocity"
ACTION_GAIN = "Gain"
ACTION_SHIFT = "Shift"
CONSUMPTION = "Consumption"
CURRENT_OPERATIONAL_COST = "CurrentOperationalCost"
EFFECTIVE_ACTION_GAIN_BETA = "EffectiveActionGainBeta"
EFFECTIVE_ACTION_VELOCITY_ALPHA = "EffectiveActionVelocityAlpha"
FATIGUE = "Fatigue"
FATIGUE_BASE = "FatigueBase"
OPERATIONAL_COSTS_CONV = "OperationalCostsConv"
MIS_CALIBRATION = "MisCalibration"
MIS_CALIBRATION_DOMAIN = "MisCalibrationDomain"
MIS_CALIBRATION_SYSTEM_RESPONSE = "MisCalibrationSystemResponse"
MIS_CALIBRATION_PHI_IDX = "MisCalibrationPhiIdx"
RANDOM_SEED = "RandomSeed"
REWARD_TOTAL = "RewardTotal"
SET_POINT = "SetPoint"

LONG_TO_SHORT = collections.OrderedDict()
LONG_TO_SHORT[ACTION_VELOCITY] = 'v'
LONG_TO_SHORT[ACTION_GAIN] = 'g'
LONG_TO_SHORT[ACTION_SHIFT] = 's'
LONG_TO_SHORT[CONSUMPTION] = 'c'
LONG_TO_SHORT[CURRENT_OPERATIONAL_COST] = 'coc'
LONG_TO_SHORT[EFFECTIVE_ACTION_GAIN_BETA] = 'ge'
LONG_TO_SHORT[EFFECTIVE_ACTION_VELOCITY_ALPHA] = 've'
LONG_TO_SHORT[FATIGUE] = 'f'
LONG_TO_SHORT[FATIGUE_BASE] = 'fb'
LONG_TO_SHORT[OPERATIONAL_COSTS_CONV] = 'oc'
LONG_TO_SHORT[MIS_CALIBRATION] = 'MC'
LONG_TO_SHORT[MIS_CALIBRATION_DOMAIN] = 'gs_domain'
LONG_TO_SHORT[MIS_CALIBRATION_PHI_IDX] = 'gs_phi_idx'
LONG_TO_SHORT[MIS_CALIBRATION_SYSTEM_RESPONSE] = 'gs_sys_response'
LONG_TO_SHORT[RANDOM_SEED] = 'seed'
LONG_TO_SHORT[REWARD_TOTAL] = 'cost'
LONG_TO_SHORT[SET_POINT] = 'p'
# These i could not link to anything (in Java industrialbenchmark),
# but they sound not particularly useful/important.. right? ;-)
#self.state['o'] = np.zeros(10) #  operational cost buffer
#self.state['hg'] = 0. # hidden gain
#self.state['hv'] = 0. # hidden velocity
#self.state['hs'] = 0. # hidden shift

SHORT_TO_LONG = collections.OrderedDict()
for l in LONG_TO_SHORT.keys():
	SHORT_TO_LONG[LONG_TO_SHORT[l]] = l

class IndustrialBenchmarkEnv(gym.Env):
	'''An OpenAI Gym environment implementation/wrapper
	of the (Siemens) Industrial Benchmark simulator of gas and wind turbines
	(Python version).

	To install prerequisites:
		# Python 3
		sudo pip3 install Numpy
		sudo pip3 install JPype1-py3
		sudo pip3 install gym
		sudo pip3 install scipy
		sudo pip3 install matplotlib

	See:
	* https://github.com/openai/gym/blob/master/gym/core.py
	* https://gym.openai.com/docs
	* https://github.com/siemens/industrialbenchmark
	'''

	def __init__(self, sim_props_file):
		'''Creates a new IndustrialBenchmarkEnv on the basis
		of simulation properties from a properties file.

		Keyword arguments:
		sim_props_file -- file path string to a *.properties;
				example: industrialbenchmark/src/main/resources/sim.properties
		'''

		if not sim_props_file is None:
			raise ValueError('sim_props_file is not None: no sim_props_file suport as of yet!')

		self._industrial_benchmark_dynamics = IDS()

		self._sim_props_file = sim_props_file
		# TODO loadproperties file, and set the contained values in the IDS/_industrial_benchmark_dynamics
		'''
		#self._props = self._jpkg_ind_bench_properties.PropertiesUtil.loadSetPointProperties(jpype.java.io.File(self._sim_props_file));
		#self._java_env = self._jpkg_ind_bench_dyn.IndustrialBenchmarkDynamics(self._props)
		#self._action = self._jpkg_ind_bench_data_vec_action.ActionDelta(0.0, 0.0, 0.0)
		#self._seed_data_vec = self._jpkg_ind_bench_data_vec.DataVectorImpl(to_java_list([RANDOM_SEED]))

		# Override (from gym.Env)
		#self.reward_range = (
		#		int(self._props.getProperty(ObservableStateDescription.REWARD_TOTAL + "_MIN")),
		#		int(self._props.getProperty(ObservableStateDescription.REWARD_TOTAL + "_MAX")))
		#self.action_space = from_java_list(self._action.getDescription().getVarNames()) # returns java.util.List<String>
		#self.observation_space = from_java_list(self._java_env.getMarkovState().getDescription().getVarNames()) # returns java.util.List<String>
		self.reward_range = ( -6500, 0 )
		self.action_space = [ DELTA_VELOCITY, DELTA_GAIN, DELTA_SHIFT ] # TODO?
		#self.observation_space = [ "REWARD_TOTAL", "outputPropA" ] # TODO
		#self.observation_space = self._industrial_benchmark_dynamics.markovState().keys()
		self.observation_space = LONG_TO_SHORT.keys()
		#self.observation_space = SHORT_TO_LONG.keys()
		'''
		'''
		self.reward_range = ( -6500, 0 )
		self.action_space = [ DELTA_VELOCITY, DELTA_GAIN, DELTA_SHIFT ] # TODO?
		self.observation_space = LONG_TO_SHORT.keys()
		'''

		fMin = sys.float_info.min
		fMax = sys.float_info.max

		self.reward_range = ( -6500, 0 )

		self.action_space = spaces.Tuple((
			NamedShape(DELTA_VELOCITY, spaces.Box(0.0, 100.0, 1)),
			NamedShape(DELTA_GAIN, spaces.Box(0.0, 100.0, 1)),
			NamedShape(DELTA_SHIFT, spaces.Box(0.0, 100.0, 1))
			))

		self.observation_space = spaces.Tuple((
			NamedShape(ACTION_VELOCITY, spaces.Box(fMin, fMax, 1)),
			NamedShape(ACTION_GAIN, spaces.Box(fMin, fMax, 1)),
			NamedShape(ACTION_SHIFT, spaces.Box(fMin, fMax, 1)),
			NamedShape(CONSUMPTION, spaces.Box(fMin, fMax, 1)),
			NamedShape(CURRENT_OPERATIONAL_COST, spaces.Box(fMin, fMax, 1)),
			NamedShape(EFFECTIVE_ACTION_GAIN_BETA, spaces.Box(fMin, fMax, 1)),
			NamedShape(EFFECTIVE_ACTION_VELOCITY_ALPHA, spaces.Box(fMin, fMax, 1)),
			NamedShape(FATIGUE, spaces.Box(fMin, fMax, 1)),
			NamedShape(FATIGUE_BASE, spaces.Box(fMin, fMax, 1)),
			NamedShape(OPERATIONAL_COSTS_CONV, spaces.Box(fMin, fMax, 1)),
			NamedShape(MIS_CALIBRATION, spaces.Box(fMin, fMax, 1)),
			NamedShape(MIS_CALIBRATION_DOMAIN, spaces.Box(fMin, fMax, 1)),
			NamedShape(MIS_CALIBRATION_PHI_IDX, spaces.Box(fMin, fMax, 1)),
			NamedShape(MIS_CALIBRATION_SYSTEM_RESPONSE, spaces.Box(fMin, fMax, 1)),
			NamedShape(RANDOM_SEED, spaces.Box(fMin, fMax, 1)),
			NamedShape(REWARD_TOTAL, spaces.Box(fMin, fMax, 1)),
			NamedShape(SET_POINT, spaces.Box(fMin, fMax, 1)),
			))

		self.internal_seed(int(time.time()))

	def _markov_state_to_observation(self, markov_state):

		#observation = markov_state

		#observation = dict()
		#for s in markov_state.keys():
			#if (s in SHORT_TO_LONG):
				#observation[SHORT_TO_LONG[s]] = markov_state[s]
		#observation[RANDOM_SEED] = self.current_seed

		observation = []
		for s in SHORT_TO_LONG:
			if s == LONG_TO_SHORT[RANDOM_SEED]:
				observation.append(self.current_seed)
			else:
				observation.append(markov_state[s])

		return observation

	def _step(self, action):

		#internalAction = [action[DELTA_VELOCITY], action[DELTA_GAIN], action[DELTA_SHIFT]]
		internalAction = action

		markov_state = self._industrial_benchmark_dynamics.markovState()

		reward = markov_state[LONG_TO_SHORT[REWARD_TOTAL]]

		observation = self._markov_state_to_observation(markov_state)

		done = False # never finnished
		info = None # XXX Anything useful we could put here?
		return [observation, reward, done, info]

	def _reset(self):
		self._industrial_benchmark_dynamics = IDS()

	def _render(self, mode='human', close=False):
		'''We do not support any visualization.
		'''
		return

	def _seed(self, seed=None):
		return self.internal_seed(seed)

	def internal_seed(self, seed=None):
		# Simply set the global numpy random seed,
		# as this is all that IDS is using.
		# XXX Not sure if this is good enough!
		# If it should prove to not be enough:
		# TODO FIXME extend the python industrial benchmark with an internally kept seed value
		self.current_seed = seed

		# Set Python's random seed
		random.seed(self.current_seed)

		# Set Numpy's random seed
		np.random.seed(self.current_seed)

		return [self.current_seed]
