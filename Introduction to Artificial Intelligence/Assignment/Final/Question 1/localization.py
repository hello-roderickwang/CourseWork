#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-05-11 17:24:05
# @Author  : Xuenan(Roderick) Wang
# @Email   : roderick_wang@outlook.com
# @Github  : https://github.com/hello-roderickwang

import numpy as np

class Map:
	def __init__(self):
		self.map_size = 37
		file = open('./data/Maze.txt', 'r')
		array = np.array([])
		for x in file:
			for y in x:
				if y is not '\t' and y is not '\n':
					array = np.append(array, y)
		# print('array:', array)
		if len(array) != self.map_size*self.map_size:
			print('Load file length error! File length:', len(array))
		self.map = array.reshape((self.map_size, self.map_size))
		self.print_map()
		self.get_info()

	def print_map(self):
		print('map:\n', self.map)
		print('shape of map:', self.map.shape)

	def get_info(self):
		ctr_0 = 0
		ctr_1 = 0
		ctr_error = 0
		ctr_array = np.array(['0', '1'])
		for i in range(self.map_size):
			for j in range(self.map_size):
				# print('self.map[i][j]:', self.map[i][j])
				# print('type of self.map[i][j]:', type(self.map[i][j]))
				# print('type of ctr_array[0]:', type(ctr_array[0]))
				if self.map[i][j] == ctr_array[0]:
					ctr_0 += 1
				elif self.map[i][j] == ctr_array[1]:
					ctr_1 += 1
				else:
					ctr_error += 1
					print('error:', self.map[i][j])
		print('There are', ctr_0, 'zeros and', ctr_1, 'ones in map.')
		print('There are', ctr_error, 'errors.')

class MazeSolvingBot:
	def __init__(self):
		self.map = Map()
		self.bot_position = [0, 0]

	def random_start(self):
		x = np.random.randint(1, 37)
		y = np.random.randint(1, 37)
		if self.map.map[x][y] != self.map.map[0][0]:
			self.map.map[x][y] = 'S'
			self.bot_position[0] = x
			self.bot_position[1] = y
		else:
			return self.random_start()

	def move(self, direction, step):
		for i in range(step):
			if direction is 'u':
				# print('bot_position:', self.bot_position)
				if self.map.map[int(self.bot_position[0])-1][int(self.bot_position[1])] != self.map.map[0][0]:
					self.bot_position[0] = self.bot_position[0]-1
			if direction is 'd':
				if self.map.map[int(self.bot_position[0])+1][int(self.bot_position[1])] != self.map.map[0][0]:
					self.bot_position[0] = self.bot_position[0]+1
			if direction is 'l':
				if self.map.map[int(self.bot_position[0])][int(self.bot_position[1])-1] != self.map.map[0][0]:
					self.bot_position[1] = self.bot_position[1]-1
			if direction is 'r':
				if self.map.map[int(self.bot_position[0])][int(self.bot_position[1])+1] != self.map.map[0][0]:
					self.bot_position[1] = self.bot_position[1]+1

	def is_goal(self):
		if self.map.map[self.bot_position[0]][self.bot_position[1]] == 'G':
			return True
		else:
			return False

	def get_neighbor_num(self):
		ctr = 0
		if self.map.map[int(self.bot_position[0])-1][int(self.bot_position[1])-1] == self.map.map[0][0]:
			ctr += 1
		if self.map.map[int(self.bot_position[0])][int(self.bot_position[1])-1] == self.map.map[0][0]:
			ctr += 1
		if self.map.map[int(self.bot_position[0])+1][int(self.bot_position[1])-1] == self.map.map[0][0]:
			ctr += 1
		if self.map.map[int(self.bot_position[0])+1][int(self.bot_position[1])] == self.map.map[0][0]:
			ctr += 1
		if self.map.map[int(self.bot_position[0])+1][int(self.bot_position[1])+1] == self.map.map[0][0]:
			ctr += 1
		if self.map.map[int(self.bot_position[0])][int(self.bot_position[1])+1] == self.map.map[0][0]:
			ctr += 1
		if self.map.map[int(self.bot_position[0])-1][int(self.bot_position[1])+1] == self.map.map[0][0]:
			ctr += 1
		if self.map.map[int(self.bot_position[0])-1][int(self.bot_position[1])] == self.map.map[0][0]:
			ctr += 1
		return ctr

if __name__ == '__main__':
	my_map = Map()

	bot = MazeSolvingBot()

	neighbor_array = np.array([])
	for i in range(bot.map.map_size):
		for j in range(bot.map.map_size):
			bot.bot_position = [i, j]
			if bot.map.map[i][j] == '1':
				neighbor_array = np.append(neighbor_array, -1)
			if bot.map.map[i][j] == '0':
				neighbor_array = np.append(neighbor_array, bot.get_neighbor_num())
			if bot.map.map[i][j] == 'G':
				neighbor_array = np.append(neighbor_array, -2)
	neighbor_map = neighbor_array.reshape((bot.map.map_size, bot.map.map_size))
	print('neighbor_map:', neighbor_map)
	coordinate_array = np.array([])
	ctr_5 = 0
	for i in range(bot.map.map_size):
		for j in range(bot.map.map_size):
			if neighbor_map[i][j] == 5:
				ctr_5 += 1
				coordinate_array = np.append(coordinate_array, [i, j])
	# print('coordinate for neighbor_num = 5 are:', coordinate_array)
	print('ctr_5:', ctr_5, 'len(coordinate_array):', len(coordinate_array))
	two_left_array = np.array([])
	for i in range(int(len(coordinate_array)/2)):
		x = coordinate_array[i*2]
		y = coordinate_array[i*2+1]
		bot.bot_position = [x, y]
		bot.move('l', 1)
		if bot.get_neighbor_num() == 5:
			bot.move('l', 1)
			if bot.get_neighbor_num() == 5:
				two_left_array = np.append(two_left_array, [x, y])
	# print('After two LEFT move, coordinate for neighbor_num = 5 are:', two_left_array)
	print('Number of satisfied points:', len(two_left_array)/2)

	# sequence of observation: 5, 2, 5, 5, 5, 6
	# sequence of movement: Down, Down, Right, Right, Right
	init_array = np.array([])
	target_array = np.array([])
	for i in range(int(len(coordinate_array)/2)):
		x = coordinate_array[i*2]
		y = coordinate_array[i*2+1]
		bot.bot_position = [x, y]
		bot.move('d', 1)
		if bot.get_neighbor_num() == 2:
			bot.move('d', 1)
			if bot.get_neighbor_num() == 5:
				bot.move('r', 1)
				if bot.get_neighbor_num() == 5:
					bot.move('r', 1)
					if bot.get_neighbor_num() == 5:
						bot.move('r', 1)
						if bot.get_neighbor_num() == 6:
							init_array = np.append(init_array, [x, y])
							target_array = np.append(target_array, bot.bot_position)
	print('init_array:', init_array)
	print('target_array:', target_array)
	print('number of target points:', len(target_array)/2)
