#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-02-27
# @Author  : Xuenan(Roderick) Wang
# @email   : roderick_wang@outlook.com
# @Link    : https://github.com/hello-roderickwang

import numpy as np
import matplotlib.pyplot as plt
import random

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
		print('map:', self.map)
		for i in range(self.mine_number):
			#print('Here is for-loop!')
			v_mine = np.append(v_mine, random.randint(0, self.dimention-1))
			h_mine = np.append(h_mine, random.randint(0, self.dimention-1))
		print('v_mine:', v_mine)
		print('h_mine:', h_mine)
		for i in range(0, self.mine_number):
			self.set_mine(int(v_mine[i]), int(h_mine[i]))
			#still not right, this should be return to is_mine and go forward whenever is satisfied.
		print('mine map:', self.map)

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
	#Agent should NOT have direct access to this function
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
			#x1-x3 and y1-y3
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = self.map[x+1][y-1]
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = self.map[x+1][y+1]
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = self.map[x-1][y+1]
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = self.map[x-1][y-1]
		elif x-1>=0 and x+1<=self.dimention-1 and y+1<=self.dimention-1:
			#y0 aside (x0,y0) and (x4,y0) [north side]
			neighbor['north'] = -3
			neighbor['north_east'] = -3
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = self.map[x+1][y+1]
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = self.map[x-1][y+1]
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = -3
		elif x-1>=0 and y-1>=0 and y+1<=self.dimention-1:
			#x4 aside (x4,y0) and (x4,y4) [east side]
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = -3
			neighbor['east'] = -3
			neighbor['south_east'] = -3
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = self.map[x-1][y+1]
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = self.map[x-1][y-1]
		elif x-1>=0 and x+1<=self.dimention-1 and y-1>=0:
			#y4 aside (x0,y4) and (x4,y4) [south side]
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = self.map[x+1][y-1]
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = -3
			neighbor['south'] = -3
			neighbor['south_west'] = -3
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = self.map[x-1][y-1]
		elif x+1<=self.dimention-1 and y-1>=0 and y+1<=self.dimention-1:
			#x0 aside (x0,y0) and (x0,y4) [west side]
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = self.map[x+1][y-1]
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = self.map[x+1][y+1]
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = -3
			neighbor['west'] = -3
			neighbor['north_west'] = -3
		elif x == 0 and y == 0:
			#(x0,y0)
			neighbor['north'] = -3
			neighbor['north_east'] = -3
			neighbor['east'] = self.map[x+1][y]
			neighbor['south_east'] = self.map[x+1][y+1]
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = -3
			neighbor['west'] = -3
			neighbor['north_west'] = -3
		elif x == self.dimention-1 and y == 0:
			#(x4,y0)
			neighbor['north'] = -3
			neighbor['north_east'] = -3
			neighbor['east'] = -3
			neighbor['south_east'] = -3
			neighbor['south'] = self.map[x][y+1]
			neighbor['south_west'] = self.map[x-1][y+1]
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = -3
		elif x == self.dimention-1 and y == self.dimention-1:
			#(x4,y4)
			neighbor['north'] = self.map[x][y-1]
			neighbor['north_east'] = -3
			neighbor['east'] = -3
			neighbor['south_east'] = -3
			neighbor['south'] = -3
			neighbor['south_west'] = -3
			neighbor['west'] = self.map[x-1][y]
			neighbor['north_west'] = self.map[x-1][y-1]
		elif x == 0 and y == self.dimention-1:
			#(x0,y4)
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

if __name__ == '__main__':
	my_map = Map(10, 15)
	my_map.set_ready()
	print('My Map:\n', my_map.map)