#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-05-13 20:14:00
# @Author  : Xuenan(Roderick) Wang
# @Email   : roderick_wang@outlook.com
# @Github  : https://github.com/hello-roderickwang

import numpy as np
import matplotlib.pyplot as plt
import math

class Classifier:
	def __init__(self):
		self.class_A = np.array([])
		self.class_B = np.array([])
		self.class_U = np.array([])
		self.feature_A = np.zeros((5, 4))
		self.feature_B = np.zeros((5, 4))
		self.feature_U = np.zeros((5, 4))
		self.mat_x = []
		self.mat_y = []
		self.mat_u = []

	def sigmoid(self, x):
		return 1.0/(1.0+np.exp(-x))

	def logic_regression(self, alpha = 0.01, iter = 500):
		x = self.mat_x
		y = self.mat_y
		sample_num, feature_num = np.shape(x)
		weights = np.ones((feature_num, 1))
		for i in range(iter):
			fx = x*weights
			hx = self.sigmoid(fx)
			weights += alpha*x.T*(y-hx)
		return weights

	def get_accuracy(self, weights):
		u = self.mat_u
		# y = self.mat_y
		sample_num, feature_num = np.shape(u)
		accuracy = 0.0
		for i in range(sample_num):
			predict = self.sigmoid(u[i, :]*weights)[0, 0] > 0.5
			print('(is in class A)predict:', predict)
			# if predict == bool(y[i, 0]):
			# 	accuracy += 1
		# print('accuracy:{0}%'.format(accuracy/sample_num*100))

	def load_data(self):
		file_A = np.loadtxt('./Data/ClassA.txt')
		file_B = np.loadtxt('./Data/ClassB.txt')
		file_U = np.loadtxt('./Data/Mystery.txt')
		self.class_A = np.array(np.split(file_A, 5))
		self.class_B = np.array(np.split(file_B, 5))
		self.class_U = np.array(np.split(file_U, 5))

	def normalize(self, x):
		min_x = min(x)
		max_x = max(x)
		for i in range(len(x)):
			x[i] = (x[i]-min_x)/(max_x-min_x)
		return x

	def knn(self, k):
		array_a = []
		array_b = []
		array_u = []
		for i in range(5):
			array_a.append(math.sqrt(self.feature_A[i][0]**2+self.feature_A[i][1]**2+self.feature_A[i][2]**2+self.feature_A[i][3]))
			array_b.append(math.sqrt(self.feature_B[i][0]**2+self.feature_B[i][1]**2+self.feature_B[i][2]**2+self.feature_B[i][3]))
			array_u.append(math.sqrt(self.feature_U[i][0]**2+self.feature_U[i][1]**2+self.feature_U[i][2]**2+self.feature_U[i][3]))
		print('array_a:', array_a)
		print('array_b:', array_b)
		print('array_u:', array_u)
		array = []
		for i in range(10):
			if i < 5:
				array.append(array_a[i])
			else:
				array.append(array_b[i-5])
		# print('array:',array)
		# array_n = self.normalize(array)
		# print('array:', array)
		ctr = 0
		for i in range(10):
			if array[i]<k:
				ctr += 1
		print('numbers in class A:', ctr)
		ctr_u = 0
		result_u = []
		for i in range(5):
			if array_u[i] < k:
				ctr_u += 1
				result_u.append('A')
			else:
				result_u.append('B')
		print('result_u:', result_u)

	def get_feature(self):
		for i in range(5):
			array_lu = []
			array_ru = []
			array_ld = []
			array_rd = []
			for x in range(5):
				for y in range(5):
					if self.class_A[i][x][y] == 1:
						array_lu.append(math.sqrt((x-0)**2+(y-0)**2))
						array_ru.append(math.sqrt((x-4)**2+(y-0)**2))
						array_ld.append(math.sqrt((x-0)**2+(y-4)**2))
						array_rd.append(math.sqrt((x-4)**2+(y-4)**2))
			sum = np.zeros(4)
			for j in range(len(array_lu)):
				sum[0] += array_lu[j]
				sum[1] += array_ru[j]
				sum[2] += array_ld[j]
				sum[3] += array_rd[j]
			self.feature_A[i][0] = sum[0]/len(array_lu)
			self.feature_A[i][1] = sum[1]/len(array_lu)
			self.feature_A[i][2] = sum[2]/len(array_lu)
			self.feature_A[i][3] = sum[3]/len(array_lu)
		print('self.feature_A:', self.feature_A)
		for i in range(5):
			array_lu = []
			array_ru = []
			array_ld = []
			array_rd = []
			for x in range(5):
				for y in range(5):
					if self.class_B[i][x][y] == 1:
						array_lu.append(math.sqrt((x-0)**2+(y-0)**2))
						array_ru.append(math.sqrt((x-4)**2+(y-0)**2))
						array_ld.append(math.sqrt((x-0)**2+(y-4)**2))
						array_rd.append(math.sqrt((x-4)**2+(y-4)**2))
			sum = np.zeros(4)
			for j in range(len(array_lu)):
				sum[0] += array_lu[j]
				sum[1] += array_ru[j]
				sum[2] += array_ld[j]
				sum[3] += array_rd[j]
			self.feature_B[i][0] = sum[0]/len(array_lu)
			self.feature_B[i][1] = sum[1]/len(array_lu)
			self.feature_B[i][2] = sum[2]/len(array_lu)
			self.feature_B[i][3] = sum[3]/len(array_lu)
		print('self.feature_B:', self.feature_B)
		for i in range(5):
			array_lu = []
			array_ru = []
			array_ld = []
			array_rd = []
			for x in range(5):
				for y in range(5):
					if self.class_U[i][x][y] == 1:
						array_lu.append(math.sqrt((x-0)**2+(y-0)**2))
						array_ru.append(math.sqrt((x-4)**2+(y-0)**2))
						array_ld.append(math.sqrt((x-0)**2+(y-4)**2))
						array_rd.append(math.sqrt((x-4)**2+(y-4)**2))
			sum = np.zeros(4)
			for j in range(len(array_lu)):
				sum[0] += array_lu[j]
				sum[1] += array_ru[j]
				sum[2] += array_ld[j]
				sum[3] += array_rd[j]
			self.feature_U[i][0] = sum[0]/len(array_lu)
			self.feature_U[i][1] = sum[1]/len(array_lu)
			self.feature_U[i][2] = sum[2]/len(array_lu)
			self.feature_U[i][3] = sum[3]/len(array_lu)
		print('self.feature_U:', self.feature_U)
		for i in range(5):
			self.mat_x.append(self.feature_A[i][:])
			self.mat_y.append(1)
			self.mat_x.append(self.feature_B[i][:])
			self.mat_y.append(0)
			self.mat_u.append(self.feature_U[i][:])
		self.mat_x = np.mat(self.mat_x)
		self.mat_y = np.mat(self.mat_y).T
		self.mat_u = np.mat(self.mat_u)
		print('mat_x:', self.mat_x)
		print('mat_y:', self.mat_y)

if __name__ == '__main__':
	modle = Classifier()
	# modle.get_feature()
	modle.load_data()
	modle.get_feature()
	weights = modle.logic_regression(alpha = 0.01, iter = 500)
	modle.get_accuracy(weights)
	modle.knn(5.5)