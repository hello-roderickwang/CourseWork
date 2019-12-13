#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:49:49 2019

@author: Himmel
"""

import random
import numpy as np
import matplotlib.pyplot as plt

class Map:
    def __init__(self, size):
        self.size = size;
        self.terrain = {}
        for x in range(self.size):
            for y in range(self.size):
                self.terrain[(x, y)] = 0
        self.target = (0,0)
        self.typeReport = ("", "")
        
    def generateMap(self):
        locations = list(self.terrain.keys())
        random.shuffle(locations)
        for coordinate in locations:
            probability = random.random()
            if probability <= 0.2:
                self.terrain[coordinate] = 1000 #flat
            elif probability <= 0.5:
                self.terrain[coordinate] = 800 #hilly
            elif probability <= 0.8:
                self.terrain[coordinate] = 500 #forested
            else:
                self.terrain[coordinate] = 0   #cave
        random.shuffle(locations)
        self.target = random.choice(locations)
    
    def printMap(self):
        graph = np.zeros((self.size, self.size), dtype = int)
        for (x, y) in self.terrain.keys():
            graph[x, y] = self.terrain[(x, y)]
        plt.figure(figsize=(7.5,7.5))
        plt.pcolor(graph[::-1],edgecolors='black',cmap='gist_earth', linewidths=2)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
    
    def targetMove(self):
        choice = []
        (x, y) = self.target
        if x - 1 >= 0:
            choice.append((x - 1, y))
        if y - 1 >= 0:
            choice.append((x, y - 1))
        if x + 1 < self.size:
            choice.append((x + 1, y))
        if y + 1 < self.size:
            choice.append((x, y + 1))
        self.target = random.choice(choice)
        self.typeReport = (self.terrain[(x, y)], self.terrain[self.target])
    
class SearchRobot:
    def __init__(self, Map):
        self.Map = Map
        self.belief = {} #containing belief
        self.choices = []
        self.probability = {} #finding probability
        for (x, y) in self.Map.terrain.keys():
            self.belief[(x, y)] = 1/(self.Map.size ** 2)
        self.observation = ()
        self.ptype = 0
    
    def randomPick(self):
        '''
        Choosing the locations with the highest probability of having target.
        Then take the best terrain(order: flat, hilly, forest, cave).
        Randomly picking one from the these choices
        '''
        highestProbability = max(self.belief.values())
        choices = {}
        for coordinate in self.belief.keys():
            if self.belief[coordinate] == highestProbability:
                choices[coordinate] = self.Map.terrain[coordinate]
        betterChoices = []
        terrain = max(choices.values())
        for coordinate in choices.keys():
            if choices[coordinate] == terrain:
                betterChoices.append(coordinate)
        return random.choice(list(choices.keys()))
    
    def specificPick(self):
        '''
        Choose a certain kind of area, based on the type report. 
        Pick the location with highest probability
        '''
        if self.Map.typeReport == ("",""):
            return self.randomPick()
        (origination, destination) = self.Map.typeReport
        choices = {}
        self.choices = []
        for coordinate in self.belief.keys():
            if self.Map.terrain[coordinate] == destination:
                (x, y) = coordinate
                if self.choices == []:
                    if x - 1 >= 0 and self.Map.terrain[(x - 1, y)] == origination:
                        choices[coordinate] = self.belief[coordinate]
                    elif y - 1 >= 0 and self.Map.terrain[(x, y - 1)] == origination:
                        choices[coordinate] = self.belief[coordinate]
                    elif x + 1 < self.Map.size and self.Map.terrain[(x + 1, y)] == origination:
                        choices[coordinate] = self.belief[coordinate]
                    elif y + 1 < self.Map.size and self.Map.terrain[(x, y + 1)] == origination:
                        choices[coordinate] = self.belief[coordinate]
                else:
                    if (x - 1, y) in self.choices:
                        choices[coordinate] = self.belief[coordinate]
                    elif (x, y - 1) in self.choices:
                        choices[coordinate] = self.belief[coordinate]
                    elif (x + 1, y) in self.choices:
                        choices[coordinate] = self.belief[coordinate]
                    elif (x, y + 1) in self.choices:
                        choices[coordinate] = self.belief[coordinate]
        self.choices = list(choices.keys())
        return random.choice(self.choices)
        
    
    def search(self, location):
        '''
        Pick a location with highest probability and search it.
        Give the false negative table to generate certain feeback
        '''
        (x, y) = location
        self.observation = (x, y)
        probability = random.random()
        if self.Map.terrain[(x, y)] == 1000:
            self.ptype = 0.1
            if probability <= 0.1 or (x, y) != self.Map.target:
                return "Failure"
            else:
                return "Success"
        elif self.Map.terrain[(x, y)] == 800:
            self.ptype = 0.3
            if probability <= 0.3 or (x, y) != self.Map.target:
                return "Failure"
            else:
                   return "Success"
        elif self.Map.terrain[(x, y)] == 500:
            self.ptype = 0.7
            if probability <= 0.7 or (x, y) != self.Map.target:
                return "Failure"
            else:
                return "Success"
        else:
            self.ptype = 0.9
            if probability <= 0.9 or (x, y) != self.Map.target:
                return "Failure"
            else:
                return "Success"

    def stationarySearch(self):
        '''
        If agent currently does not give us the positive(Success) feeback, keep searching
        Updating the belief of each location during the searching process
        '''
        number = 0
        while self.search(self.randomPick()) != "Success":
            denominator = 1 - self.belief[self.observation] + self.belief[self.observation] * self.ptype
            for coordinate in self.belief.keys():
                if coordinate == self.observation:
                    if self.Map.terrain[coordinate] == 1000:
                        self.belief[coordinate] *= 0.1
                    elif self.Map.terrain[coordinate] == 800:
                        self.belief[coordinate] *= 0.3
                    elif self.Map.terrain[coordinate] == 500:
                        self.belief[coordinate] *= 0.7
                    else:
                        self.belief[coordinate] *= 0.9
                self.belief[coordinate] = self.belief[coordinate]/denominator
            number += 1
        print(self.observation)
        if self.Map.terrain[self.Map.target] == 1000:
            print("Target is at flat terrain")
        elif self.Map.terrain[self.Map.target] == 800:
            print("Target is at hilly terrain")
        elif self.Map.terrain[self.Map.target] == 500:
            print("Target is at forested terrain")
        else:
            print("Target is in a cave")
        print("Congratulations! You've found the target!")
        return number
    
    def movingSearch(self):
        '''
        Each search-failure will make the target moving into a neighbor location
        The new probabilities will be updated the same as stationary search
        '''
        number = 0       
        while self.search(self.specificPick()) != "Success":
            denominator = 1 - self.belief[self.observation] + self.belief[self.observation] * self.ptype
            for coordinate in self.belief.keys():
                if coordinate == self.observation:
                    if self.Map.terrain[coordinate] == 1000:
                        self.belief[coordinate] *= 0.1
                    elif self.Map.terrain[coordinate] == 800:
                        self.belief[coordinate] *= 0.3
                    elif self.Map.terrain[coordinate] == 500:
                        self.belief[coordinate] *= 0.7
                    else:
                        self.belief[coordinate] *= 0.9
                self.belief[coordinate] = self.belief[coordinate]/denominator
            self.Map.targetMove()
            number += 1
        print(self.observation)
        if self.Map.terrain[self.Map.target] == 1000:
            print("Target is at flat terrain")
        elif self.Map.terrain[self.Map.target] == 800:
            print("Target is at hilly terrain")
        elif self.Map.terrain[self.Map.target] == 500:
            print("Target is at forested terrain")
        else:
            print("Target is in a cave")
        print("Congratulations! You've found the target!")
        return number
    
    def regression(self):
        for coordinate in self.belief.keys():
            self.belief[coordinate] = 1/(self.Map.size ** 2)
                
        
m = Map(50)
stationary = []
moving = []
'''
for i in range(50):
    m.generateMap()
    agent = SearchRobot(m)
    stationary.append(agent.stationarySearch())
    #agent.regression()
    #moving.append(agent.movingSearch())

mean= sum(stationary) // 50
std = 0
for number in stationary:
    std += (number - mean) ** 2
std = (std / 50)** 0.5

normalDistribution = np.random.normal(mean, std, 100000)
plt.hist(normalDistribution, bins=100, normed=True)
plt.show()
plt.xlabel("the order of experiment")
plt.ylabel("the number of steps to find target")
plt.plot(range(1,51), stationary)
'''
m.generateMap()
for i in range(100):
    agent = SearchRobot(m)
#stationary.append(agent.stationarySearch())
    moving.append(agent.movingSearch())    
mean = sum(moving) // 100
std = 0
for number in moving:
    std += (number - mean) ** 2
std = (std / 100)** 0.5
normalDistribution = np.random.normal(mean, std, 100000)
plt.hist(normalDistribution, bins=100, normed=True)
plt.show()

plt.xlabel("the order of experiment")
plt.ylabel("the number of steps to find target")
plt.plot(range(1,101), moving)
