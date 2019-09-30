#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-02-27
# @Author  : Xuenan(Roderick) Wang
# @email   : roderick_wang@outlook.com
# @Link    : https://github.com/hello-roderickwang

import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.setrecursionlimit(1000000)

class Map:
	def __init__(self, d:'dimention of Map', n:'number of mines'):
		self.dimention = d
		self.mine_number = n
		self.map = np.zeros((self.dimention, self.dimention))

	def generate_map(self):
		v_mine = np.array([])
		h_mine = np.array([])
		print('self.dimention:', self.dimention)
		print('self.mine_number:', self.mine_number)
		print('map:\n', self.map)
		for i in range(self.mine_number):
			# print('Here is for-loop!')
			v_mine = np.append(v_mine, random.randint(0, self.dimention-1))
			h_mine = np.append(h_mine, random.randint(0, self.dimention-1))
		print('v_mine:', v_mine)
		print('h_mine:', h_mine)
		for i in range(0, self.mine_number):
			self.set_mine(int(v_mine[i]), int(h_mine[i]))
			# still not right, this should be return to is_mine and go forward whenever is satisfied.
		print('mine map:\n', self.map)

	def is_mine(self, x:'x coordinate', y:'y coordinate'):
		if self.map[x][y] == -1:
			return True
		else:
			return False

	def set_mine(self, x:'x coordinate', y:'y coordinate'):
		if self.is_mine(x, y) == False:
			self.map[x][y] = -1
		elif self.is_mine(x, y) == True:
			random_x = random.randint(0, self.dimention-1)
			random_y = random.randint(0, self.dimention-1)
			return self.set_mine(random_x, random_y)
		else:
			print('There is an error in return of is_mine function!')

	def go_east(self, x:'x coordinate', y:'y coordinate'):
		return x+1, y

	def go_west(self, x:'x coordinate', y:'y coordinate'):
		return x-1, y

	def go_north(self, x:'x coordinate', y:'y coordinate'):
		return x, y-1

	def go_south(self, x:'x coordinate', y:'y coordinate'):
		return x, y+1

	def get_neighbor(self, x:'x coordinate', y:'y coordinate') -> 'return a dictionary in which contains map relation of direction and box value':
	# Agent should NOT have direct access to this function
		"""
		sample map
		@  -3 **out of map**
		@  -2 **signal for mine**
		@  -1 **mine**
		@   0 **no mine or neighbor no mine**
		@ 1-8 **number of neignbor mines**

		   x0 x1 x2 x3 x4 outside
		 y0 1 -1 -1  2  1 -3
		 y1 2  3  4 -1  1 -3
		 y2 1 -1  3  3  2 -3
		 y3 1  2 -1  2 -1 -3
		 y4 0  1  1  2  1 -3
		   -3 -3 -3 -3 -3 -3
		"""
		neighbor = {'north' : -3, 'north_east' : -3, 'east' : -3, 'south_east' : -3, 'south' : -3, 'south_west' : -3, 'west' : -3, 'north_west' : -3}
		if x-1>=0 and x+1<=self.dimention-1 and y-1>=0 and y+1<=self.dimention-1:
			# x1-x3 and y1-y3
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = self.map[x+1][y-1]
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = self.map[x+1][y+1]
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = self.map[x-1][y+1]
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = self.map[x-1][y-1]
		elif x-1>=0 and x+1<=self.dimention-1 and y+1<=self.dimention-1:
			# y0 aside (x0,y0) and (x4,y0) [north side]
			neighbor['north'] = -3
			neighbor['north_east'] = -3
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = self.map[x+1][y+1]
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = self.map[x-1][y+1]
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = -3
		elif x-1>=0 and y-1>=0 and y+1<=self.dimention-1:
			# x4 aside (x4,y0) and (x4,y4) [east side]
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = -3
			neighbor['east'] = -3
			neighbor['south_east'] = -3
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = self.map[x-1][y+1]
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = self.map[x-1][y-1]
		elif x-1>=0 and x+1<=self.dimention-1 and y-1>=0:
			# y4 aside (x0,y4) and (x4,y4) [south side]
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = self.map[x+1][y-1]
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = -3
			neighbor['south'] = -3
			neighbor['south_west'] = -3
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = self.map[x-1][y-1]
		elif x+1<=self.dimention-1 and y-1>=0 and y+1<=self.dimention-1:
			# x0 aside (x0,y0) and (x0,y4) [west side]
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = self.map[x+1][y-1]
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = self.map[x+1][y+1]
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = -3
			neighbor['west'] = -3
			neighbor['north_west'] = -3
		elif x == 0 and y == 0:
			# (x0,y0)
			neighbor['north'] = -3
			neighbor['north_east'] = -3
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = self.map[x+1][y+1]
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = -3
			neighbor['west'] = -3
			neighbor['north_west'] = -3
		elif x == self.dimention-1 and y == 0:
			# (x4,y0)
			neighbor['north'] = -3
			neighbor['north_east'] = -3
			neighbor['east'] = -3
			neighbor['south_east'] = -3
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = self.map[x-1][y+1]
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = -3
		elif x == self.dimention-1 and y == self.dimention-1:
			# (x4,y4)
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = -3
			neighbor['east'] = -3
			neighbor['south_east'] = -3
			neighbor['south'] = -3
			neighbor['south_west'] = -3
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = self.map[x-1][y-1]
		elif x == 0 and y == self.dimention-1:
			# (x0,y4)
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = self.map[x+1][y-1]
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = -3
			neighbor['south'] = -3
			neighbor['south_west'] = -3
			neighbor['west'] = -3
			neighbor['north_west'] = -3
		else:
			print('There is an error in Map.get_neighbor function!')
		return neighbor

	def get_box_value(self, x:'x coordinate', y:'y coordinate') -> 'return an integer which is the mine number of neighborhood':
		neighbor = self.get_neighbor(x, y)
		box_mine_number = 0
		for direction in neighbor:
			if neighbor[direction] == -1:
				box_mine_number += 1
		return box_mine_number

	def is_mine(self, x:'x coordinate', y:'y coordinate'):
		if self.map[x][y] == -1:
			return True
		elif self.map[x][y] >= 0:
			return False
		else:
			print('This is an illegal access!')

	def set_ready(self):
		self.generate_map()
		for y in range(0, self.dimention):
			for x in range(0, self.dimention):
				if self.is_mine(x, y) == False:
					self.map[x][y] = self.get_box_value(x, y)
				elif self.is_mine(x, y) == True:
					self.map[x][y] = -1
				else:
					print('Problem in function of set_value!')

class PlayGround():
	def __init__(self, d:'dimention of Map', n:'number of mines', t:'threshold for mines'):
		# dimention of PlayGround must keep same as Map
		self.dimention = d
		self.mine_number = n
		self.play_ground = np.zeros((self.dimention, self.dimention))
		self.play_ground.fill(-2)
		self.mine_map = Map(self.dimention, self.mine_number)
		self.mine_map.set_ready()
		self.next_step = [-1, -1]
		self.wrong_step = 0
		self.mine_threshold = t
		print('play_ground:\n', self.play_ground)
		"""
		sample play_ground
		@  -3 **out of map**
		@  -2 **origin state**
		@  -1 **flag a mine**
		@ 0-8 **number in the same coordinate in mine_map**

		   x0 x1 x2 x3 x4 outside
		 y0 1 -1 -1  2  1 -3
		 y1 2  3  4 -1  1 -3
		 y2 1 -1  3  3  2 -3
		 y3 1  2 -1  2 -1 -3
		 y4 0  1  1  2  1 -3
		   -3 -3 -3 -3 -3 -3
		"""

	def make_guess(self) -> 'return random coordinate x, y':
		x = random.randint(0, self.dimention-1)
		y = random.randint(0, self.dimention-1)
		print('guess coordinate is: (', x, ', ', y, ')')
		if self.is_visited(x, y) == True:
			return self.make_guess()
		else:
			return [x, y]

	def expand_box(self, x:'x coordinate', y:'y coordinate'):
		'''
		self.play_ground[x-1][y-1] = self.mine_map[x-1][y-1]
			self.play_ground[x][y-1] = self.mine_map[x][y-1]
			self.play_ground[x+1][y-1] = self.mine_map[x+1][y-1]
			self.play_ground[x-1][y] = self.mine_map[x-1][y]
			self.play_ground[x+1][y] = self.mine_map[x+1][y]
			self.play_ground[x-1][y+1] = self.mine_map[x-1][y+1]
			self.play_ground[x][y+1] = self.mine_map[x][y+1]
			self.play_ground[x+1][y+1] = self.mine_map[x+1][y+1]
			'''
		if self.is_zero(x-1, y-1) == True:
			self.expand_box(x-1, y-1)
		elif self.is_zero(x, y-1) == True:
			self.expand_box(x, y-1)
		elif self.is_zero(x+1, y-1) == True:
			self.expand_box(x+1, y-1)
		elif self.is_zero(x-1, y) == True:
			self.expand_box(x-1, y)
		elif self.is_zero(x+1, y) == True:
			self.expand_box(x+1, y)
		elif self.is_zero(x-1, y+1) == True:
			self.expand_box(x-1, y+1)
		elif self.is_zero(x, y+1) == True:
			self.expand_box(x, y+1)
		elif self.is_zero(x+1, y+1) == True:
			self.expand_box(x+1, y+1)

	'''
	def is_expandable(self, x:'x coordinate', y:'y coordinate'):
		if self.mine_map[x][y] == 0:
			return True
		else:
			return False
	'''

	def is_visited(self, x:'x coordinate', y:'y coordinate'):
		if x < 0 or x >= self.dimention:
			return False
		if y < 0 or y >= self.dimention:
			return False
		if self.play_ground[x][y] >= -1 and self.play_ground[x][y] <= 0:
			return True
		else:
			return False

	def is_mine(self, x:'x coordinate', y:'y coordinate'):
		if x < 0 or x >= self.dimention:
			return False
		if y < 0 or y >= self.dimention:
			return False
		if self.mine_map.map[x][y] == -1:
			return True
		else:
			return False

	def is_zero(self, x:'x coordinate', y:'y coordinate'):
		if x < 0 or x >= self.dimention:
			return False
		if y < 0 or y >= self.dimention:
			return False
		if self.mine_map.map[x][y] == 0:
			self.play_ground[x][y] = 0
			return True
		else:
			return False

	def count_not_visited_neighbor(self, x:'x coordinate', y:'y coordinate'):
		i = 0
		# list_available_neighbor = np.array([[],[]])
		available_neighbor_x = np.array([])
		available_neighbor_y = np.array([])
		'''
		if self.is_visited(x-1, y-1) == True:
			i += 1
			list_available_neighbor = np.append(list_available_neighbor[:][0], x-1)
			list_available_neighbor = np.append(list_available_neighbor[:][1], y-1)
		if self.is_visited(x, y-1) == True:
			i += 1
			list_available_neighbor = np.append(list_available_neighbor[:][0], x)
			list_available_neighbor = np.append(list_available_neighbor[:][1], y-1)
		if self.is_visited(x+1, y-1) == True:
			i += 1
			list_available_neighbor = np.append(list_available_neighbor[:][0], x+1)
			list_available_neighbor = np.append(list_available_neighbor[:][1], y-1)
		if self.is_visited(x-1, y) == True:
			i += 1
			list_available_neighbor = np.append(list_available_neighbor[:][0], x-1)
			list_available_neighbor = np.append(list_available_neighbor[:][1], y)
		if self.is_visited(x+1, y) == True:
			i += 1
			list_available_neighbor = np.append(list_available_neighbor[:][0], x+1)
			list_available_neighbor = np.append(list_available_neighbor[:][1], y)
		if self.is_visited(x-1, y+1) == True:
			i += 1
			list_available_neighbor = np.append(list_available_neighbor[:][0], x-1)
			list_available_neighbor = np.append(list_available_neighbor[:][1], y+1)
		if self.is_visited(x, y+1) == True:
			i += 1
			list_available_neighbor = np.append(list_available_neighbor[:][0], x)
			list_available_neighbor = np.append(list_available_neighbor[:][1], y+1)
		if self.is_visited(x+1, y+1) == True:
			i += 1
			list_available_neighbor = np.append(list_available_neighbor[:][0], x+1)
			list_available_neighbor = np.append(list_available_neighbor[:][1], y+1)
		return i, list_available_neighbor
		'''
		if self.is_visited(x-1, y-1) == True:
			i += 1
			available_neighbor_x = np.append(available_neighbor_x, x-1)
			available_neighbor_y = np.append(available_neighbor_y, y-1)
		if self.is_visited(x, y-1) == True:
			i += 1
			available_neighbor_x = np.append(available_neighbor_x, x)
			available_neighbor_y = np.append(available_neighbor_y, y-1)
		if self.is_visited(x+1, y-1) == True:
			i += 1
			available_neighbor_x = np.append(available_neighbor_x, x+1)
			available_neighbor_y = np.append(available_neighbor_y, y-1)
		if self.is_visited(x-1, y) == True:
			i += 1
			available_neighbor = np.append(available_neighbor_x, x-1)
			available_neighbor = np.append(available_neighbor_y, y)
		if self.is_visited(x+1, y) == True:
			i += 1
			available_neighbor = np.append(available_neighbor_x, x+1)
			available_neighbor = np.append(available_neighbor_y, y)
		if self.is_visited(x-1, y+1) == True:
			i += 1
			available_neighbor = np.append(available_neighbor_x, x-1)
			available_neighbor = np.append(available_neighbor_y, y+1)
		if self.is_visited(x, y+1) == True:
			i += 1
			available_neighbor = np.append(available_neighbor_x, x)
			available_neighbor = np.append(available_neighbor_y, y+1)
		if self.is_visited(x+1, y+1) == True:
			i += 1
			available_neighbor = np.append(available_neighbor_x, x+1)
			available_neighbor = np.append(available_neighbor_y, y+1)
		return i, available_neighbor_x, available_neighbor_y

	'''
	def make_move(self, x:'x coordinate', y:'y coordinate'):
		if self.is_mine(x, y) == True:
			print('You LOSE!! Try again!!')
		elif self.is_zero(x, y) == True:
			self.expand_box(x, y)
		else:
	'''

	# conditional probability - P(p_new|p_pre)
	def calculate_possibility(self, p_new:'possibility of new event', p_pre:'possibility of origin'):
		# return ((p_new+p_pre)-p_new*p_pre)/p_pre
		return 1-((1-p_new)*(1-p_pre))

	def label_possibility(self, x:'x coordinate', y:'y coordinate'):
		if self.is_zero(x, y) == False:
			if self.is_mine(x, y) == False:
				# this bos is visited and testified that is not a mine, therefore the possibility of this box being a mine is 0
				self.play_ground[x][y] = 0
				visited, available_neighbor_x, available_neighbor_y = self.count_not_visited_neighbor(x, y)
				for neighbor_x in range(0, len(available_neighbor_x)):
					for neighbor_y in range(0, len(available_neighbor_y)):
						if self.is_visited(neighbor_x, neighbor_y) == False:
							self.play_ground[neighbor_x][neighbor_y] = self.mine_map.map[x][y]/visited
						else:
							self.play_ground[neighbor_x][neighbor_y] = self.calculate_possibility(self.mine_map.map[x][y]/visited, self.play_ground[neighbor_x][neighbor_y])


				'''
				# below is wrong
				if self.is_visited(x, y) == True:
					self.play_ground[x][y] = self.calculate_possibility(self.mine_map[x][y]/(8-visited), self.play_ground[x][y])
				else:
					self.play_ground[x][y] = self.mine_map[x][y]/(8-visited)
				'''

	def label_mine(self, x:'x coordinate', y:'y coordinate'):
		x = int(x)
		y = int(y)
		self.play_ground[x][y] = -1

	def sort_possibility(self):
		least = np.zeros(3)
		largest = np.zeros(3)
		for y in range(0, self.dimention):
			for x in range(0, self.dimention):
				if self.play_ground[x][y] < least[0]:
					least[0] = self.play_ground[x][y]
					least[1] = x
					least[2] = y
				elif self.play_ground[x][y] > largest[0]:
					largest[0] = self.play_ground[x][y]
					largest[1] = x
					largest[2] = y
		return least, largest

	def reach_mine_threshold(self, largest):
		if largest[0] >= self.mine_threshold:
			self.label_mine(largest[1], largest[2])
			return True
		else:
			return False

	def is_game_end(self):
		i = 0
		for y in range(0, self.dimention):
			for x in range(0, self.dimention):
				if self.play_ground[x][y] == 0 or self.play_ground[x][y] == -1:
					i += 1
		if i == self.dimention * self.dimention:
			return True
		else:
			return False

	def run(self, x:'x coordinate', y:'y coordinate'):
		x = int(x)
		y = int(y)
		print('this time coordinate: (', x, ', ', y, ')')
		print('now play_ground is:\n', self.play_ground)
		if self.is_game_end() == True:
			print('mine_map:\n', self.mine_map.map)
			print('play_ground:\n', self.play_ground)
		else:
			if self.is_mine(x, y) == True:
				self.label_mine(x, y)
				new_x, new_y = self.make_guess()
				return self.run(new_x, new_y)
			elif self.is_zero(x, y) == True:
				self.expand_box(x, y)
				least, largest = self.sort_possibility()
				if self.reach_mine_threshold(largest) == True:
					self.label_mine(largest[1], largest[2])
				return self.run(least[1], least[2])
			else:
				self.label_possibility(x, y)
				least, largest = self.sort_possibility()
				if self.reach_mine_threshold(largest) == True:
					self.label_mine(largest[1], largest[2])
				return self.run(least[1], least[2])

if __name__ == '__main__':
	'''
	my_map = Map(10, 15)
	my_map.set_ready()
	print('My Map:\n', my_map.map)
	'''
	my_board = PlayGround(5, 5, 0.8)
	x, y = my_board.make_guess()
	my_board.run(x, y)





