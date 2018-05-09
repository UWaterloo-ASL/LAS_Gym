"""defines pre-scripted behaviours"""
import time
import random
from math import *
from Parameter import *


class Behaviour:

	def __init__(self, joint_handles):
		# constants
		self.idle_wait_time = 4
		self.idle_gap = 3.0
		self.propagate_gap = 2.0
		self.light_duration = 4 # unit: second
		self.sma_duration = 4 # unit:second

		self.state = 0 # set it to idle at first
		self.active_list = []
		self.propagate_list = []
		self.idle_event_start_time = 0.0
		self.active_event_start_time = 0.0

		self.sma_handles = joint_handles
		self.parameter_list = []
		for handle in self.sma_handles:
			self.parameter_list.append(Parameter('sma'))

	def idle_random_event(self):
		# print('time is ' + str(type(time.time())))
		# print('idle_event_start_time is ' + str(self.idle_event_start_time))
		# print('idle_gap is ' + str(type(idle_gap)))
		if time.time() - self.idle_event_start_time > self.idle_gap:

			r_index = random.randint(0, len(self.sma_handles) - 1)
			print('random index = ' + str(r_index))
			# rand_patch_par = patch_parameter_list[rand_patch_key]
			if not self.parameter_list[r_index].busy:
				self.active_list.append(r_index)
				print("active list:")
				print(self.active_list)
			# update idle start time
			self.idle_event_start_time = time.time()

		for idx in self.active_list:
			finish = self._single_motion(self.sma_handles[idx], self.parameter_list[idx])
			if finish:
				self.active_list.remove(idx)

	def _single_motion(self, actr, actr_parameter):

		# perform one step for actuator
		finish = False
		actr_type = actr_parameter.gettype()

		if not actr_parameter.busy:
			actr_parameter.actrstart()

		if actr_type == 'sma':
			t = time.time() - actr_parameter.start_time
			# print('t=' + str(t))
			if t < self.sma_duration:
				# sma_index = 1
				# target = 1.5 # simulate the magnitude of controls current
				v = sin(t / 10) * (t / 10)
				actr.set_matrix([0, 0, 0, 0, 0, 0, 0, 0, v, 0, 0, 0]) # <<<<<<<<<<<<<<<<<<<<< change here
			else:
				v = 0
				actr.set_matrix(
					[0, 0, 0, 0, 0, 0, 0, 0, v, 0, 0, 0])
				finish = True
				actr_parameter.actrstop()

		# if time.time() - actr_parameter.start_time < light_duration:

		return finish

	def active_event(self, trigger_list):

		if len(self.propagate_list) == 0:
			# assume only one sensor is triggered each time
			if len(trigger_list) > 0:
				self._gen_propagate(trigger_list[0], self.sma_handles)

		if time.time() - self.active_event_start_time > self.propagate_gap:

			self.active_event_start_time = time.time()

			if len(self.propagate_list) > 0:
				next_group = self.propagate_list.popleft()
				for i in next_group:
					if not self.parameter_list[i].busy:
						self.active_list.append(i)
						print("active list:")
						print(self.active_list)

		for idx in self.active_list:
			finish = self._single_motion(self.sma_handles[idx], self.parameter_list[idx])
			if finish:
				self.active_list.remove(idx)

	def _gen_propagate(self, index, sma_handles):
		"""
		:param index: Trigger centre
		:param sma_handles: All sma handles
		:return:
		"""

		'''
		Might need to change to propagate according to XYZ coordinates
		'''
		reach_left = False
		reach_right = False
		self.propagate_list.append(index)
		while (not reach_left) and (not reach_right):
			last_group = self.propagate_list[-1]
			next_group = []
			if not reach_left:
				if last_group[0] == 0:
					reach_left = True
				else:
					next_group.append(last_group[0] - 1)
			if not reach_right:
				if last_group[1] == len(sma_handles)-1:
					reach_right = True
				else:
					next_group.append(last_group[1] + 1)
			self.propagate_list.append(next_group)

	def idle_start(self):
		self.idle_event_start_time = time.time()
		self.state = 0

	def active_start(self):
		self.active_event_start_time = time.time()
		self.state = 1

	def read_sensor(self, observation):
		trigger_list = []
		# go through observations

		return trigger_list

	def lasg_behaviour(self, observation):
		trigger_ls = self.read_sensor(observation) # <<<<<<<<<<<<<<<<<<<< need observation here ??
		if len(trigger_ls) == 0 and time.time() - self.idle_event_start_time >= self.idle_wait_time:
			# Idle state
			# Update start time only at state transitions
			if self.state != 0:
				self.idle_start()
				print('state = ' + str(self.state))
		elif len(trigger_ls) > 0:
			# Active state
			# activate blocks
			if self.state != 1:
				self.active_start()
				print('state = ' + str(self.state))

		if self.state == 0:
			self.idle_random_event()

		if self.state == 1:
			# propagate the action
			# assume ONLY ONE sensor is triggered each time
			self.active_event(trigger_ls)

	#
	#
	# def find_location(self, patch_map, trigger_key):
	# 	# Find coordinate [row, col] of triggered patch
	# 	coordinate = [-1, -1]
	# 	for m in range(len(patch_map)):
	# 		for n in range(len(patch_map[0])):
	# 			if patch_map[m][n] == trigger_key:
	# 				coordinate = [m, n]
	#
	# 	return coordinate