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
indBenEnv = IndustrialBenchmarkEnv.IndustrialBenchmarkEnv(sim_props_file)

print("\nindBenEnv.reward_range:")
print(indBenEnv.reward_range)
print("")

print("\nindBenEnv.action_space:")
#print("\tlen: %d" % len(indBenEnv.action_space))
print(indBenEnv.action_space)
print("\tsample:")
print(indBenEnv.action_space.sample())
print("")

print("\nindBenEnv.observation_space:")
#print("\tlen: %d" % len(indBenEnv.observation_space))
print(indBenEnv.observation_space)
print("\tsample:")
print(indBenEnv.observation_space.sample())
print("")

# Execute a single step
action = (
	0.1, # IndustrialBenchmarkEnv.DELTA_VELOCITY
	0.1, # IndustrialBenchmarkEnv.DELTA_GAIN
	0.1 # IndustrialBenchmarkEnv.DELTA_SHIFT
	)
[observation, reward, done, info] = indBenEnv.step(action)

print("\nobservation (len: %d):" % len(observation))
print(observation)
print("\nreward:")
print(reward)
print("\ndone:")
print(done)
print("\ninfo:")
print(info)
print("")

# Some more testing
[observation, reward, done, info] = indBenEnv.step(action)

indBenEnv.seed(12345)
[observation, reward, done, info] = indBenEnv.step(action)

indBenEnv.seed(123456)
indBenEnv.reset()
[observation, reward, done, info] = indBenEnv.step(action)
