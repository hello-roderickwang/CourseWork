#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date	: 2019-04-24 16:56:02

import numpy as np
import math
import sys
import copy
from itertools import permutations
import time
from statistics import mean
import matplotlib.pyplot as plt


class Node:
    def __init__(self):
        self.weight = 0
        # Node.type defination
        # type = -1  ---- starting point
        # type = 0   ---- normal point(unvisited)
        # type = 1   ---- normal point(visited)
        self.type = 0
        self.name = 'unknown'

    def set_weight(self, weight):
        self.weight = weight

    def get_name(self):
        return self.name

    def get_weight(self):
        return self.weight

    def set_type(self, type):
        if type is 'visited':
            self.type = 1
        elif type is 'unvisited':
            self.type = 0
        elif type is 'start':
            self.type = -1
        else:
            print('UNKNOWN type of node, please check!')

    def get_type(self):
        return self.type


class Map(Node):
    def __init__(self, num=np.random.randint(5, 10), map_type=0):
        self.node_num = num
        self.edge_matrix = np.zeros((self.node_num, self.node_num))
        self.node_array = []
        # Map.map_type defination
        # map_type = 0 ---- without node weight
        # map_type = 1 ---- with node weight
        self.map_type = map_type
        self.node_total_weight = 0
        self.node_visited_weight = 0
        self.total_visited_dist = 0
        self.last_weight_array = []
        self.generate_edge()
        self.generate_node()
        self.select_start()
        if self.map_type is 1:
            self.get_node_total_weight()

    def reload(self, map_type=0, weight_reload=False):
        if self.map_type is 1:
            if len(self.last_weight_array) is 0:
                for i in range(self.node_num):
                    self.last_weight_array.append(self.node_array[i].weight)
            else:
                for i in range(self.node_num):
                    self.last_weight_array[i] = self.node_array[i].weight
        # print('self.last_weight_array:', self.last_weight_array)
        self.map_type = map_type
        self.node_visited_weight = 0
        for n in range(self.node_num):
            if n is 0:
                self.node_array[n].set_type('start')
            else:
                self.node_array[n].set_type('unvisited')
        if self.map_type is 1:
            if weight_reload is False and len(self.last_weight_array) is not 0:
                for i in range(self.node_num):
                    self.node_array[i].weight = self.last_weight_array[i]
            elif weight_reload is True or len(self.last_weight_array) is 0:
                for i in range(self.node_num):
                    self.node_array[i].weight = np.random.randint(1, 5)
            else:
                print('ERROR! weight_load illegal!')
            self.get_node_total_weight()
        else:
            for i in range(self.node_num):
                self.node_array[i].weight = 0
            self.get_node_total_weight()

    def set_map(self, matrix):
        self.node_num = len(matrix[0])
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i is j:
                    self.edge_matrix[i][j] = 0
                    self.node_array[i].weight = matrix[i][j]
                    self.last_weight_array.append(matrix[i][j])
                self.edge_matrix[i][j] = matrix[i][j]

    def get_node_influence(self):
        if self.map_type is 0:
            return 1
        else:
            return 2 - self.node_visited_weight / self.node_total_weight

    def get_node_total_weight(self):
        self.node_array[0].weight = 0
        self.node_total_weight = 0
        for i in range(1, self.node_num):
            self.node_total_weight += self.node_array[i].weight

    def generate_node(self):
        if self.map_type is 0:
            for i in range(self.node_num):
                self.node_array.append(Node())
        elif self.map_type is 1:
            for i in range(self.node_num):
                self.node_array.append(Node())
                self.node_array[i].weight = np.random.randint(1, 5)

    def generate_edge(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i is not j:
                    self.edge_matrix[i][j] = np.random.randint(1, 10)

    def print_node(self):
        if self.map_type is 0:
            print('Map.node_array:\n', self.node_array)
        elif self.map_type is 1:
            print('Map.node_array:\n', self.node_array)
            for i in range(self.node_num):
                print('Map.node_array[', i, ']:',
                      self.node_array[i].weight, '\n')

    def print_edge(self):
        print('Map.edge_matrix:\n', self.edge_matrix)

    def print_node_weight(self):
        weight_a = []
        for i in range(self.node_num):
            weight_a.append(self.node_array[i].weight)
        print('node_weight:', weight_a)

    def select_start(self):
        # n_start = np.random.randint(0, self.node_num)
        n_start = 0
        self.node_array[n_start].set_type('start')
        self.node_array[n_start].weight = 0

    def explore(self, node):
        return self.node_array[node].type, self.node_array[node].weight


class BruteForce:
    def __init__(self, target_map):
        self.map = target_map
        self.permutations_matrix = np.zeros(
            (math.factorial(self.map.node_num - 1), self.map.node_num - 1))
        self.cost_array = []
        self.minimum_cost = 0
        self.ratio = 1
        self.optimized_sequence = []

    def set_ratio(self, ratio):
        self.ratio = ratio

    def run(self):
        self.permutations_list = list(
            permutations(range(1, self.map.node_num)))
        print('self.permutations_list:', self.permutations_list)
        num = self.map.node_num
        for i in range(len(self.permutations_list)):
            cost = 0
            self.map.node_visited_weight = 0
            cost += self.map.get_node_influence() * \
                self.map.edge_matrix[0][self.permutations_list[i][0]]
            for j in range(num - 2):
                # print('length of tuple:', len(self.permutations_list[i]))
                # print('self.map.edge_matrix:[', self.permutations_list[i][j], '][', self.permutations_list[i][j+1], ']')
                self.map.node_visited_weight += self.map.node_array[self.permutations_list[i][j]].weight
                # print('node_visited_weight:', self.map.node_visited_weight)
                cost += self.map.get_node_influence() * \
                    self.map.edge_matrix[self.permutations_list[i]
                                         [j]][self.permutations_list[i][j + 1]]
            self.map.node_visited_weight += self.map.node_array[self.permutations_list[i][num - 2]].weight
            # print('node_visited_weight:', self.map.node_visited_weight)
            cost += self.map.edge_matrix[self.permutations_list[i][num - 2]][0]
            self.cost_array.append(cost)
        print('cost_array:', self.cost_array)
        self.minimum_cost = min(self.cost_array)
        for i in range(len(self.cost_array)):
            if self.minimum_cost is self.cost_array[i]:
                for j in range(num - 1):
                    self.optimized_sequence.append(
                        self.permutations_list[i][j])

    def get_result(self):
        return self.minimum_cost, self.optimized_sequence


class Dynamic:
    def __init__(self, target_map):
        self.map = target_map
        self.total_cost = 0
        self.optimized_sequence = []

    def run(self):
        Matrix = self.map.edge_matrix
        # Here Matrix represents the graph of TSP. Matrix[i][j] is the distance when i<>j and is the weight when i==j
        n = len(Matrix)
        for i in range(1, n):
            Matrix[i][i] = self.map.node_array[i].weight
        state = []
        new = [(1, [1], 0.0, 0.0)]
        state.append(new)
        end = 0
        visited = []
        cost = 0.0
        weight = 0.0
        point = list(range(1, n + 1))
        for j in range(0, n - 1):
            new = []
            state.append(new)
            for k in range(len(state[j])):
                end = state[j][k][0]
                visited = state[j][k][1]
                cost = state[j][k][2]
                weight = state[j][k][3]
                for nextend in point:
                    if nextend not in visited:
                        nextvisited = copy.deepcopy(visited)
                        if (self.map.node_total_weight == 0):
                            nextcost = cost + Matrix[end - 1][nextend - 1]
                        else:
                            nextcost = cost + \
                                Matrix[end - 1][nextend - 1] * \
                                (2 - weight / self.map.node_total_weight)
                        nextweight = weight + Matrix[nextend - 1][nextend - 1]
                        nextvisited.append(nextend)
                        new = [nextend, nextvisited, nextcost, nextweight]
                        q = 0
                        for m in range(len(state[j + 1])):
                            if sorted(nextvisited) == sorted(state[j + 1][m][1]) and nextend == state[j + 1][m][0]:
                                q = 1
                                if nextcost < state[j + 1][m][2]:
                                    state[j + 1][m][2] = nextcost
                                    state[j + 1][m][1] = nextvisited
                        if q == 0:
                            state[j + 1].append(new)
        j = n - 1
        new = []
        state.append(new)
        for k in range(len(state[j])):
            end = state[j][k][0]
            visited = state[j][k][1]
            nextvisited = copy.deepcopy(visited)
            cost = state[j][k][2]
            weight = state[j][k][3]
            nextvisited.append(1)
            if (self.map.node_total_weight == 0):
                totalcost = cost + Matrix[end - 1][0]
            else:
                totalcost = cost + Matrix[end - 1][0] * \
                    (2 - weight / self.map.node_total_weight)
            new = [end, nextvisited, totalcost, weight]
            state[j + 1].append(new)
        path = []
        Mincost = float("inf")
        j = n
        for k in range(len(state[j])):
            visited = state[j][k][1]
            cost = state[j][k][2]
            weight = state[j][k][3]
            if cost < Mincost:
                Mincost = cost
                path = visited
                Totalweight = weight
        for i in range(1, len(path) - 1):
            self.optimized_sequence.append(path[i] - 1)
        self.total_cost = Mincost

    def get_result(self):
        return self.total_cost, self.optimized_sequence


class Greedy:
    def __init__(self, target_map, sigma=0):
        self.map = target_map
        self.sigma = sigma
        self.total_cost = 0
        self.total_eva_cost = 0
        self.optimized_sequence = []

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def d_over_w(self, weight, dist):
        if self.sigma is 0:
            return self.sigmoid(6 * (weight / (weight + dist)) - 0.5)
        elif self.sigma < 1 and self.sigma > 0:
            return self.sigma
        else:
            print('ERROR! Illegal sigma!')

    def next_node(self, node_pos):
        cost_array = []
        i_array = []
        self.map.node_array[node_pos].set_type('visited')
        for i in range(self.map.node_num):
            # print('self.map.node_array[i].type:', self.map.node_array[i].type)
            if self.map.node_array[i].type is 0:
                if (self.map.node_total_weight == 0):
                    cost_array.append(self.map.edge_matrix[node_pos][i])
                else:
                    cost_array.append((2 - self.map.node_visited_weight / self.map.node_total_weight) *
                                      self.map.edge_matrix[node_pos][i] + self.map.node_array[i].weight * self.map.total_visited_dist * (self.map.node_num - i - 2) / ((i + 1) * self.map.node_total_weight))
                i_array.append(i)
        # print('cost_array:', cost_array)
        # print('i_array:', i_array)
        min_cost = cost_array[0]
        # print('initial min_cost:', min_cost)
        min_pos = i_array[0]
        for m in range(len(cost_array)):
            # print('m:', m)
            if cost_array[m] < min_cost:
                # print('go in comparison')
                min_cost = cost_array[m]
                min_pos = i_array[m]
        # print('min_cost:', min_cost)
        # print('min_pos:', min_pos)
        self.total_eva_cost += min_cost
        self.optimized_sequence.append(min_pos)
        current_length = len(self.optimized_sequence)
        self.map.total_visited_dist += self.map.edge_matrix[self.optimized_sequence[current_length - 2]][min_pos]
        self.map.node_visited_weight += self.map.node_array[min_pos].weight

    def calculate_cost(self):
        self.map.node_visited_weight = 0
        ctr = 0
        for i in range(self.map.node_num):
            if i is 0:
                self.total_cost += self.map.get_node_influence() * \
                    self.map.edge_matrix[0][self.optimized_sequence[i]]
                ctr += 1
            elif i is self.map.node_num - 1:
                self.map.node_visited_weight += self.map.node_array[self.optimized_sequence[i - 1]].weight
                self.total_cost += self.map.get_node_influence() * \
                    self.map.edge_matrix[self.optimized_sequence[i - 1]][0]
                ctr += 1
                # print('node_visited_weight:', self.map.node_visited_weight)
                # print('node_total_weight:', self.map.node_total_weight)
                # print('ctr:', ctr)
            else:
                self.map.node_visited_weight += self.map.node_array[self.optimized_sequence[i - 1]].weight
                self.total_cost += self.map.get_node_influence() * \
                    self.map.edge_matrix[self.optimized_sequence[i - 1]
                                         ][self.optimized_sequence[i]]
                ctr += 1
                # print('node_visited_weight:', self.map.node_visited_weight)

    def run(self):
        for i in range(self.map.node_num - 1):
            if i is 0:
                self.next_node(0)
            else:
                self.next_node(self.optimized_sequence[-1])
        self.calculate_cost()

    def get_result(self):
        return self.total_cost, self.optimized_sequence

    def get_evaluation(self, cost_array):
        ctr = 0
        for i in range(len(cost_array)):
            if self.total_cost > cost_array[i]:
                ctr += 1
        return ctr / len(cost_array)


if __name__ == '__main__':
    MyMap = Map(num=6, map_type=0)

    matrix = [
        [0, 3, 2.2, 1.6, 4.5, 1.8],
        [2.4, 1000, 1.6, 3.8, 5.2, 3.9],
        [1.6, 1.7, 640, 2.4, 3.9, 2.6],
        [2.2, 3.5, 3.2, 120, 2.5, 0.6],
        [2.6, 3.9, 4.8, 2.7, 560, 2.9],
        [1.8, 3.4, 2.9, 0.6, 2.7, 80]]

    MyMap.set_map(matrix)
    MyMap.print_edge()
    MyMap.print_node()

    print('-----------------------------------------------------------')
    alg_bf1 = BruteForce(MyMap)
    MyMap.print_node_weight()
    alg_bf1.run()
    cost_bf1, s_bf1 = alg_bf1.get_result()
    print('*********Brute Force********')
    print('minimum cost without node weights:', cost_bf1)
    print('optimized sequence is:', s_bf1)
    print('-----------------------------------------------------------')
    MyMap.reload(map_type=1)
    MyMap.print_node_weight()
    alg_bf2 = BruteForce(MyMap)
    alg_bf2.run()
    cost_bf2, s_bf2 = alg_bf2.get_result()
    print('*********Brute Force********')
    print('minimum cost with node weights:', cost_bf2)
    print('optimized sequence is:', s_bf2)
    print('-----------------------------------------------------------')
    MyMap.reload(map_type=0)
    MyMap.print_node_weight()
    alg_d1 = Dynamic(MyMap)
    alg_d1.run()
    cost_d1, s_d1 = alg_d1.get_result()
    print('*********Dynamic********')
    print('minimum cost without node weights:', cost_d1)
    print('optimized sequence is:', s_d1)
    print('-----------------------------------------------------------')
    MyMap.reload(map_type=1)
    MyMap.print_node_weight()
    alg_d2 = Dynamic(MyMap)
    alg_d2.run()
    cost_d2, s_d2 = alg_d2.get_result()
    print('*********Dynamic********')
    print('minimum cost without node weights:', cost_d2)
    print('optimized sequence is:', s_d2)
    print('-----------------------------------------------------------')
    MyMap.reload(map_type=0)
    MyMap.print_node_weight()
    alg_g1 = Greedy(MyMap)
    alg_g1.run()
    cost_g1, s_g1 = alg_g1.get_result()
    eva_g1 = alg_g1.get_evaluation(alg_bf1.cost_array)
    print('*********Greeedy********')
    print('minimum cost without node weights:', cost_g1)
    print('optimized sequence is:', s_g1)
    print('evaluation is:', eva_g1)
    print('-----------------------------------------------------------')
    MyMap.reload(map_type=1)
    MyMap.print_node_weight()
    alg_g2 = Greedy(MyMap)
    alg_g2.run()
    cost_g2, s_g2 = alg_g2.get_result()
    eva_g2 = alg_g2.get_evaluation(alg_bf2.cost_array)
    print('*********Greeedy********')
    print('minimum cost with node weights:', cost_g2)
    print('optimized sequence is:', s_g2)
    print('evaluation is:', eva_g2)

    # node_num_min = 3
    # node_num_max = 10
    # loop = 20
    # total_time_BF_0_array = np.zeros(node_num_max - node_num_min + 1)
    # total_time_BF_1_array = np.zeros(node_num_max - node_num_min + 1)
    # total_time_D_0_array = np.zeros(node_num_max - node_num_min + 1)
    # total_time_D_1_array = np.zeros(node_num_max - node_num_min + 1)
    # total_time_G_0_array = np.zeros(node_num_max - node_num_min + 1)
    # total_time_G_1_array = np.zeros(node_num_max - node_num_min + 1)
    # total_cost_BF_0_array = np.zeros(node_num_max - node_num_min + 1)
    # total_cost_BF_1_array = np.zeros(node_num_max - node_num_min + 1)
    # total_cost_D_0_array = np.zeros(node_num_max - node_num_min + 1)
    # total_cost_D_1_array = np.zeros(node_num_max - node_num_min + 1)
    # total_cost_G_0_array = np.zeros(node_num_max - node_num_min + 1)
    # total_cost_G_1_array = np.zeros(node_num_max - node_num_min + 1)
    # total_eva_G_0_array = np.zeros(node_num_max - node_num_min + 1)
    # total_eva_G_1_array = np.zeros(node_num_max - node_num_min + 1)
    # for i in range(node_num_max - node_num_min + 1):
    # 	time_BF_0_array = []
    # 	time_BF_1_array = []
    # 	time_D_0_array = []
    # 	time_D_1_array = []
    # 	time_G_0_array = []
    # 	time_G_1_array = []
    # 	cost_BF_0_array = []
    # 	cost_BF_1_array = []
    # 	cost_D_0_array = []
    # 	cost_D_1_array = []
    # 	cost_G_0_array = []
    # 	cost_G_1_array = []
    # 	eva_G_0_array = []
    # 	eva_G_1_array = []
    # 	for j in range(loop):
    # 		# initialize the map
    # 		test_map = Map(num = i + node_num_min, map_type = 0)
    # 		# BruteForce map.type = 0
    # 		start_BF_0 = time.time()
    # 		alg_BF_0 = BruteForce(test_map)
    # 		test_map.print_node_weight()
    # 		alg_BF_0.run()
    # 		cost_BF_0, s_BF_0 = alg_BF_0.get_result()
    # 		end_BF_0 = time.time()
    # 		# BruteForce map.type = 1
    # 		test_map.reload(map_type = 1)
    # 		start_BF_1 = time.time()
    # 		alg_BF_1 = BruteForce(test_map)
    # 		test_map.print_node_weight()
    # 		alg_BF_1.run()
    # 		cost_BF_1, s_BF_1 = alg_BF_1.get_result()
    # 		end_BF_1 = time.time()
    # 		# Dynamic map.type = 0
    # 		test_map.reload(map_type = 0)
    # 		start_D_0 = time.time()
    # 		alg_D_0 = Dynamic(test_map)
    # 		test_map.print_node_weight()
    # 		alg_D_0.run()
    # 		cost_D_0, s_D_0 = alg_D_0.get_result()
    # 		end_D_0 = time.time()
    # 		# Dynamic map.type = 1
    # 		test_map.reload(map_type = 1)
    # 		start_D_1 = time.time()
    # 		alg_D_1 = Dynamic(test_map)
    # 		test_map.print_node_weight()
    # 		alg_D_1.run()
    # 		cost_D_1, s_D_1 = alg_D_1.get_result()
    # 		end_D_1 = time.time()
    # 		# Greedy map.type = 0
    # 		test_map.reload(map_type = 0)
    # 		start_G_0 = time.time()
    # 		alg_G_0 = Greedy(test_map)
    # 		test_map.print_node_weight()
    # 		alg_G_0.run()
    # 		cost_G_0, s_G_0 = alg_G_0.get_result()
    # 		end_G_0 = time.time()
    # 		eva_G_0 = alg_G_0.get_evaluation(alg_BF_0.cost_array)
    # 		# Greedy map.type = 1
    # 		test_map.reload(map_type = 1)
    # 		start_G_1 = time.time()
    # 		alg_G_1 = Greedy(test_map)
    # 		test_map.print_node_weight()
    # 		alg_G_1.run()
    # 		cost_G_1, s_G_1 = alg_G_1.get_result()
    # 		end_G_1 = time.time()
    # 		eva_G_1 = alg_G_1.get_evaluation(alg_BF_1.cost_array)
    # 		# record
    # 		time_BF_0 = int(round(end_BF_0 * 1000000)) - int(round(start_BF_0 * 1000000))
    # 		time_BF_1 = int(round(end_BF_1 * 1000000)) - int(round(start_BF_1 * 1000000))
    # 		time_D_0 = int(round(end_D_0 * 1000000)) - int(round(start_D_0 * 1000000))
    # 		time_D_1 = int(round(end_D_1 * 1000000)) - int(round(start_D_1 * 1000000))
    # 		time_G_0 = int(round(end_G_0 * 1000000)) - int(round(start_G_0 * 1000000))
    # 		time_G_1 = int(round(end_G_1 * 1000000)) - int(round(start_G_1 * 1000000))
    # 		time_BF_0_array.append(math.log(time_BF_0))
    # 		time_BF_1_array.append(math.log(time_BF_1))
    # 		time_D_0_array.append(math.log(time_D_0))
    # 		time_D_1_array.append(math.log(time_D_1))
    # 		time_G_0_array.append(math.log(time_G_0))
    # 		time_G_1_array.append(math.log(time_G_1))
    # 		cost_BF_0_array.append(cost_BF_0)
    # 		cost_BF_1_array.append(cost_BF_1)
    # 		cost_D_0_array.append(cost_D_0)
    # 		cost_D_1_array.append(cost_D_1)
    # 		cost_G_0_array.append(cost_G_0)
    # 		cost_G_1_array.append(cost_G_1)
    # 		eva_G_0_array.append(eva_G_0)
    # 		eva_G_1_array.append(eva_G_1)
    # 	total_time_BF_0_array[i] = mean(time_BF_0_array)
    # 	total_time_BF_1_array[i] = mean(time_BF_1_array)
    # 	total_time_D_0_array[i] = mean(time_D_0_array)
    # 	total_time_D_1_array[i] = mean(time_D_1_array)
    # 	total_time_G_0_array[i] = mean(time_G_0_array)
    # 	total_time_G_1_array[i] = mean(time_G_1_array)
    # 	total_cost_BF_0_array[i] = mean(cost_BF_0_array)
    # 	total_cost_BF_1_array[i] = mean(cost_BF_1_array)
    # 	total_cost_D_0_array[i] = mean(cost_D_0_array)
    # 	total_cost_D_1_array[i] = mean(cost_D_1_array)
    # 	total_cost_G_0_array[i] = mean(cost_G_0_array)
    # 	total_cost_G_1_array[i] = mean(cost_G_1_array)
    # 	total_eva_G_0_array[i] = mean(eva_G_0_array)
    # 	total_eva_G_1_array[i] = mean(eva_G_1_array)
    # print('np.arange(node_num_min, node_num_max + 1):', np.arange(node_num_min, node_num_max + 1), 'type:', type(np.arange(node_num_min, node_num_max + 1)))
    # print('total_cost_BF_0_array:', total_cost_BF_0_array, 'type:', type(total_cost_BF_0_array))
    # print('np.arange(node_num_min, node_num_max + 1).shape:', np.arange(node_num_min, node_num_max + 1).shape)
    # print('total_cost_BF_0_array.shape:', total_cost_BF_0_array.shape)
    # plt.figure(1)
    # plt.subplot(221)
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_cost_BF_0_array, color = 'blue', label = 'Brute Force')
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_cost_D_0_array, color = 'red', label = 'Dynamic Programming')
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_cost_G_0_array, color = 'green', label = 'Greedy')
    # plt.legend()
    # plt.grid(axis = 'y')
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_cost_BF_0_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_cost_D_0_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_cost_G_0_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # plt.ylabel('Average Cost')
    # plt.xlabel('Node Number of Graph')
    # plt.title('Average Costs of Different Node Numbers(without node weights)')
    # plt.subplot(222)
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_time_BF_0_array, color = 'blue', label = 'Brute Force')
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_time_D_0_array, color = 'red', label = 'Dynamic Programming')
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_time_G_0_array, color = 'green', label = 'Greedy')
    # plt.legend()
    # plt.grid(axis = 'y')
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_time_BF_0_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_time_D_0_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_time_G_0_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # plt.ylabel('Average log(Time) (ns)')
    # plt.xlabel('Node Number of Graph')
    # plt.title('Average Time of Different Node Numbers(without node weights)')
    # plt.subplot(223)
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_cost_BF_1_array, color = 'blue', label = 'Brute Force')
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_cost_D_1_array, color = 'red', label = 'Dynamic Programming')
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_cost_G_1_array, color = 'green', label = 'Greedy')
    # plt.legend()
    # plt.grid(axis = 'y')
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_cost_BF_1_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_cost_D_1_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_cost_G_1_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # plt.ylabel('Average Cost')
    # plt.xlabel('Node Number of Graph')
    # plt.title('Average Costs of Different Node Numbers(with node weights)')
    # plt.subplot(224)
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_time_BF_1_array, color = 'blue', label = 'Brute Force')
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_time_D_1_array, color = 'red', label = 'Dynamic Programming')
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_time_G_1_array, color = 'green', label = 'Greedy')
    # plt.legend()
    # plt.grid(axis = 'y')
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_time_BF_1_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_time_D_1_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # for a, b in zip(np.arange(node_num_min, node_num_max + 1), total_time_G_1_array):
    # 	plt.text(a, b+0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 7)
    # plt.ylabel('Average log(Time) (ns)')
    # plt.xlabel('Node Number of Graph')
    # plt.title('Average Time of Different Node Numbers(with node weights)')
    # # plot evaluation of greedy
    # plt.figure(2)
    # plt.subplot(121)
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_eva_G_0_array, color = 'green', label = 'Greedy')
    # plt.legend()
    # plt.grid(axis = 'y')
    # plt.ylabel('Average Percentage of Optimization')
    # plt.xlabel('Node Number of Graph')
    # plt.title('Average Percentage of Optimization of Different Node Numbers(without node weights)')
    # plt.subplot(122)
    # plt.plot(np.arange(node_num_min, node_num_max + 1), total_eva_G_1_array, color = 'green', label = 'Greedy')
    # plt.legend()
    # plt.grid(axis = 'y')
    # plt.ylabel('Average Percentage of Optimization')
    # plt.xlabel('Node Number of Graph')
    # plt.title('Average Percentage of Optimization of Different Node Numbers(with node weights)')
    # plt.show()
