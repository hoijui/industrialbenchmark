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

import gym

class NamedShape(gym.Space):
	"""
	Adds a name to a shape.
	Example usage:
	self.action_space = (
		NamedShape("dimension_x", spaces.Box(low=-10, high=10, shape=(1,)))
		NamedShape("dimension_y", spaces.Box(low=-20, high=20, shape=(1,)))
		)
	"""
	def __init__(self, name, space):
		self.space = space
		self.name = name

	def sample(self):
		return self.space.sample()
	def contains(self, x):
		return self.space.contains(self, x)

	def to_jsonable(self, sample_n):
		return self.space.to_jsonable(sample_n)
	def from_jsonable(self, sample_n):
		return self.space.from_jsonable(sample_n)

	@property
	def __repr__(self):
		return 'NamedShape(name=%s, space=%s)' % (self.name, self.space.__repr__())
	def __str__(self):
		return self.__repr__
	def __eq__(self, other):
		return self.space.__eq__(self, other)
