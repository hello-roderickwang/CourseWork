#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-05-13 04:40:23
# @Author  : Xuenan(Roderick) Wang
# @Email   : roderick_wang@outlook.com
# @Github  : https://github.com/hello-roderickwang

import numpy as np
import math

class Robot:
	def __init__(self):
		# type 0:new 1-8:used1-8 9:dead
		self.type = 0
		self.value = 0.0
		self.beta = 0.9
		self.step = 0

	def draw_lots(self, prob):
		n = np.random.randint(0, 100)
		if n/100 <= prob:
			return True
		else:
			return False

	def use(self):
		if self.type != 9:
			if self.type == 0:
				self.value += 100*math.pow(self.beta, self.step)
				self.type = 1
				self.step += 1
			elif self.type <= 7:
				if self.draw_lots(self.type*0.1) == True:
					self.value += (100-10*self.type)*math.pow(self.beta, self.step)
					self.type += 1
					self.step += 1
				else:
					self.value += (100-10*self.type)*math.pow(self.beta, self.step)
					self.step += 1
			else:
				if self.draw_lots(0.8) == True:
					self.value += 20*math.pow(self.beta, self.step)
					self.type = 9
					self.step += 1
				else:
					self.value += 20*math.pow(self.beta, self.step)
					self.step += 1

	def replace(self):
		if self.type != 0:
			self.type = 0
			self.value += -250*math.pow(self.beta, self.step)
			self.step += 1

	def predict(self, type, utility, price = 250, last_step = 0):
		self.type = type
		step = last_step
		reward = 0
		# print('self.type:', self.type)
		if self.type <= 8:
			if utility == 'use':
				reward += (100-10*self.type)*math.pow(self.beta, step)
				i = self.type+1
				step += 1
				while i <= 8:
					if self.draw_lots(0.1*i) == True:
						reward += (100-10*self.type)*math.pow(self.beta, step)
						i += 1
						step += 1
					else:
						reward += (100-10*self.type)*math.pow(self.beta, step)
						step += 1
				if i == 9:
					reward += -1*price*math.pow(self.beta, step)
			elif utility == 'replace':
				reward += -250
		return reward

	def buy_used(self):
		if self.draw_lots(0.5) == True:
			return 1
		else:
			return 2

	def used_predict(self, type, utility, price = 250):
		self.type = type
		step = 0
		reward = 0
		if self.type <= 8:
			if utility == 'use':
				reward += (100-10*self.type)*math.pow(self.beta, step)
				i = self.type+1
				step += 1
				while i <= 8:
					if self.draw_lots(0.1*i) == True:
						reward += (100-10*self.type)*math.pow(self.beta, step)
						i += 1
						step += 1
					else:
						reward += (100-10*self.type)*math.pow(self.beta, step)
						step += 1
				if i == 9:
					reward_used = self.predict(self.buy_used(), 'use', price = price, last_step = step)-1*price*math.pow(self.beta, step)
					reward_new = self.predict(0, 'use', last_step = step)-250*math.pow(self.beta, step)
			elif utility == 'replace':
				reward += -250
		return reward_used-reward_new





if __name__ == '__main__':
	bot = Robot()
	# stete: New
	# bot.use()
	# state: Used 1
	# reward_use = []
	# for i in range(9):
	# 	reward_use.append(bot.predict(i, 'use'))
	# reward_replace = []
	# for i in range(1, 10):
	# 	reward_replace.append(bot.predict(i, 'replace'))
	# print('reward_use:', reward_use)
	# print('reward_replace:', reward_replace)
	loop = 1000
	# reward_use = [0,0,0,0,0,0,0,0,0,0]
	# for i in range(loop):
	# 	for j in range(9):
	# 		reward_use[j] += bot.predict(j, 'use')
	# for i in range(9):
	# 	reward_use[i] = reward_use[i]/loop
	# print('average reward_use over 1000:', reward_use)
	# bot.predict(0, 'use')-bot.used_predict(0, 'use', j)
	# reward_used = []
	# for i in range(loop):
	# 	for j in range(0, 100):
	# 		if i == 0:
	# 			reward_used.append(bot.used_predict(8, 'use', j))
	# 		else:
	# 			reward_used[j] += bot.used_predict(8, 'use', j)
	# for i in range(len(reward_used)):
	# 	reward_used[i] = reward_used[i]/loop
	# 	if reward_used[i] >= 0:
	# 		print('i:', i)
	# print('average reward_used over 1000:', reward_used)
	# best_pos = []
	# for p in range(0, 250):
	# 	reward_use = [0,0,0,0,0,0,0,0,0,0]
	# 	pos = []
	# 	for i in range(loop):
	# 		for j in range(9):
	# 			reward_use[j] += bot.predict(j, 'use', price = p)
	# 	for i in range(9):
	# 		reward_use[i] = reward_use[i]/loop
	# 		if reward_use[i] >= 0:
	# 			pos.append(i)
	# 	best_pos.append(pos[-1])
	# print('best_pos:', best_pos)
	# array = []
	# for n in range(len(best_pos)):
	# 	if best_pos[n] > 5:
	# 		array.append(n)
	# print('accpted price:', array)
	# print('average reward_use over 1000:', reward_use)
	# bot.predict(0, 'use')-bot.used_predict(0, 'use', j)
	# best_pos = []
	# for b in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
	# 	reward_use = [0,0,0,0,0,0,0,0,0]
	# 	bot.beta = b
	# 	pos = []
	# 	for i in range(loop):
	# 		for j in range(9):
	# 			reward_use[j] += bot.predict(j, 'use')
	# 	for i in range(9):
	# 		reward_use[i] = reward_use[i]/loop
	# 		if reward_use[i] >= 0:
	# 			pos.append(i)
	# 	best_pos.append(pos[-1])
	# 	print('reward_use:', reward_use)
	# print('best_pos:', best_pos)
	cost = 0
	price = 250
	while cost >= 0:
		reward = 0
		for i in range(loop):
			reward += bot.predict(0, 'use', price = price)
		cost = reward/loop
		price += 1
	print('price that no longer being meaningful:', price)
	


