#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:58:55 2019

@author: Himmel
"""
import structure
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from datetime import datetime

e = math.e

ctr = 0

def manhattanDistance(state, goal):
        (x1, y1), (x2, y2) = state, goal
        return abs(x2 - x1) + abs(y2 - y1)
    
def euclideanDistance(state, goal):
        (x1, y1), (x2, y2) = state, goal
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2)**0.5

class Maze:
    def __init__(self, length, width, initialP):
        self.length = length
        self.width = width
        self.maze = {}
        self.initialP = initialP
        self.numerator = -(self.length + self.width)
        self.denominator = length * width -2
        self.probability = initialP * (e)**(self.numerator/self.denominator)
        for i in range(length):
            for j in range(width):
                self.maze[(i, j)] = -1
        self.start = (0, 0)
        self.goal = (length - 1, width - 1)
        
    def generateMaze(self):
        '''
            (x, y) -> 0 : empty
            (x, y) -> 1 : obstruction
        '''
        for coordinates in self.maze.keys():
            if coordinates != self.start and coordinates != self.goal:
                p = random.random()
                if p < self.probability:
                    self.maze[coordinates] = 1
                    self.denominator -= 1
                else:
                    self.maze[coordinates] = 0
                    self.numerator += 1
            self.probability = self.initialP * (e)**(self.numerator/self.denominator)
    
            
    def printMaze(self, Search, heuristic = None):
        path = self.getPath(Search, heuristic)
        mazeMap = np.zeros((self.length, self.width), dtype = int)
        for (x, y) in self.maze.keys():
            mazeMap[x, y] = self.maze[(x, y)]
        for (x, y) in path:
            mazeMap[x, y] = -1
        plt.figure(figsize=(5,5))
        plt.pcolor(mazeMap[::-1],edgecolors='black',cmap='Blues', linewidths=2)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
        
    def isWall(self, state):
        return self.maze[state] == 1
        
    def getSuccessor(self, state):
        '''input:
            - state: a tuple stands for coordinates
           output:
            - a list of successor,
            - a successor contains the neighbor's cooridinates and the action 
              for current state to go there, and the cost(distance) between
        '''
        successors = []
        (x, y) = state
        if x - 1 >= 0 and not self.isWall((x - 1, y)):
            successors.append(((x - 1, y), "west", 1))
        if y - 1 >= 0 and not self.isWall((x, y - 1)):
            successors.append(((x, y - 1), "north", 1))
        if x + 1 < self.length and not self.isWall((x + 1, y)):
            successors.append(((x + 1, y), "east", 1))
        if y + 1 < self.width and not self.isWall((x, y + 1)):
            successors.append(((x, y + 1), "south", 1))
        return successors
        
    def isGoalState(self, state):
        '''
            check if the current state is the goal state
        '''
        return state == self.goal
    
    
    def getPath(self, Search, heuristic = None):
        '''
           Input: a search function, which may have heuristic function 
           
           Turning the series of actions into coordinates 
           
           Output: a list of states               
        '''
        path = []
        if heuristic == None:
            path = Search(self)
        else:
            path = Search(self, heuristic)
        states = [self.start]
        for action in path:
            (x, y) = states[-1]
            if action == "west":
                states.append((x - 1, y))
            elif action == "north":
                states.append((x, y - 1))
            elif action == "east":
                states.append((x + 1, y))
            elif action == "south":
                states.append((x, y + 1))
        return states
    
    def simplify(self, fraction):
        '''
           input: 
            fraction: a number ∈ [0,1]
            fraction == 1: empty maze
            fraction == 0: no changes
           output:
            a copy of the maze, which strip out a fraction of the obstructions
        '''
        
        simple = Maze(self.length, self.width, 0)
        simple.maze = self.maze.copy()
        wallState = []
        numberOfWall = 0
        for state in self.maze.keys():
            if self.maze[state] == 1:
                wallState.append(state)
                numberOfWall += 1
        random.shuffle(wallState)
        stripNumber = math.ceil(fraction * numberOfWall)
        for state in wallState:
            simple.maze[state] = 0
            stripNumber -= 1
            if stripNumber == 0:
                break
        return simple
 
class FireMaze(Maze):
    def __init__(self,length, width, initialP, fire):
        '''
            initialize the number of the fire spots
        '''
        Maze.__init__(self, length, width, initialP)
        self.fire = fire
        self.fireSpots = []
        self.probabilityDistribution = {}
        for state in self.maze.keys():
            self.probabilityDistribution[state] = 0
        self.utility = {}
    
    def setFire(self):
        '''
            randomly distribute the fire spots
        '''
        coordinates = []
        for x in range((self.length // 4) + 1, 3 * self.length // 4):
            for y in range((self.width // 4) + 1, 3 * self.width // 4):
                coordinates.append((x, y))
        fire = self.fire 
        random.shuffle(coordinates)
        for coordinate in coordinates:
            self.maze[coordinate] = 2
            self.probabilityDistribution[coordinate] = 1
            self.fireSpots.append(coordinate)
            fire -= 1
            if fire == 0:
                break
            
    def fireProbability(self, state):
        '''
            Input:
                - a state coordinate
            Output:
                - a number between 0 ~ 1 indicates the probability of the state
                  changing to fire state
        '''
        
        (x, y) = state
        if self.maze[state] == 2:
            self.probabilityDistribution[state] = 1
            if state not in self.fireSpots:
                self.fireSpots.append(state)
            return 1
        '''
        probFire = 0
        if x - 1 >= 0:
            probFire += self.probabilityDistribution[(x - 1, y)]
        if y - 1 >= 0:
            probFire += self.probabilityDistribution[(x, y - 1)]
        if x + 1 < self.length:
            probFire += self.probabilityDistribution[(x + 1, y)]
        if y + 1 < self.width:
            probFire += self.probabilityDistribution[(x, y + 1)]
        return (0.25)**(probFire)
        '''
        distances = []
        for spot in self.fireSpots:
            distances.append(manhattanDistance(spot, state))
        distance = min(distances)
        return 0.25 ** (distance)
    
    def getNeighbor(self, state):
        '''
            Input: a given state with (x, y) as its coordinates
            
            Output:
                    the neighborhoods in four directions(n, e, s, w)
        '''
        (x, y) = state
        neighbor = []
        if x - 1 >= 0:
            neighbor.append((x - 1, y))
        if x + 1 < self.length:
            neighbor.append((x + 1, y))
        if y - 1 >= 0:
            neighbor.append((x, y - 1))
        if y + 1 < self.width:
            neighbor.append((x, y + 1))
        return neighbor

    def distributeFire(self):
        '''
            Distribute the fire probability to whole maze
            Using a dictionary to keep memorize the probability in each state
            Update probability when we spread fire
        '''
        probabilityLevel = self.fireSpots.copy()
        #print(probabilityLevel)
        #return None
        newProbability = {}
        for state in self.fireSpots:
            newProbability[state] = 1
        #n = 10
        for state in self.probabilityDistribution.keys():
            if self.probabilityDistribution[state] != 1:
                self.probabilityDistribution[state] = 0
        while len(newProbability) < len(self.probabilityDistribution):
            amount = len(probabilityLevel)
            for i in range(amount):
                state = probabilityLevel[i]
                #print('state:', state, 'neighbor:', self.getNeighbor(state))
                #print('current state ', state, 'has probability', self.probabilityDistribution[state])
                for neighbor in self.getNeighbor(state):
                    #print('neighbor', neighbor, 'has probability', self.probabilityDistribution[neighbor])
                    if self.probabilityDistribution[neighbor] < self.probabilityDistribution[state] and neighbor not in newProbability.keys():
                        newProbability[neighbor] = self.fireProbability(neighbor)
                        probabilityLevel.append(neighbor)
                        #print('changed neighbor:', neighbor,'now has probabilit', self.probabilityDistribution[neighbor] ,'but will have probability:', newProbability[neighbor])
            for location in newProbability.keys():
                self.probabilityDistribution[location] = newProbability[location]
            probabilityLevel = probabilityLevel[amount:]
            #print(self.probabilityDistribution)
            
            
            
        
    def spreadFire(self):
        '''
            Spread fire from the fire spots
        '''
        #print(1)
        #print(self.probabilityDistribution)
        newSpots = []
        for state in self.fireSpots:
            for neighbor in self.getNeighbor(state):
                if neighbor not in self.fireSpots and random.random() <= self.probabilityDistribution[neighbor]:
                    self.probabilityDistribution[neighbor] = 1
                    if neighbor not in newSpots:
                        newSpots.append(neighbor)
        self.fireSpots += newSpots            
        #print(1)
        #print(self.probabilityDistribution)
    
        
        
        

def DFS(maze):
        '''
            Using stack as the data structure
            return when we pop the goal state
            Output:
                a series of actions
        '''
        stack = structure.Stack()
        stack.push((maze.start, []))
        visited = {}
        global ctr
        ctr = 0
        while not stack.isEmpty():
            (state, path) = stack.pop()
            ctr += 1
            if maze.isGoalState(state):
                #print("Reach the goal!")
                return path
            for successor in maze.getSuccessor(state):
                (neighbor, action, _) = successor
                if neighbor not in visited.keys():
                    stack.push((neighbor, path + [action]))
            if state not in visited.keys():
                visited[state] = True
            if stack.isEmpty():
                #print("There is no such a path!")
                return path
                
def BFS(maze):
        '''
            Using queue as the data structure
            return when we meet the goal state
            Output:
                a series of actions
        '''
        queue = structure.Queue()
        queue.enqueue((maze.start, []))
        visited = {}
        global ctr
        ctr = 0
        while not queue.isEmpty():
            (state, path) = queue.dequeue()
            ctr += 1
            for successor in maze.getSuccessor(state):
                (neighbor, action, _) = successor
                if maze.isGoalState(neighbor):
                    #print("Reach the goal!")
                    return path + [action]
                elif neighbor not in visited.keys():
                    queue.enqueue((neighbor, path + [action]))
            if state not in visited.keys():
                visited[state] = True
            if queue.isEmpty():
                #print("There is no such a path!")
                return path
    
    
    
def Astar(maze, heuristic):
        '''
           input:
             - heuristic : a function which gives us the estimated value
           Using min-heap (priority queue) as data structure
           return when we pop the goal state
           Output:
                a series of actions
        '''
        Heap = structure.PriorityQueue()
        Heap.heap = [(0, 1, (maze.start, [], heuristic(maze.start, maze.goal)))]
        visited = {}
        global ctr
        ctr = 0
        while not Heap.isEmpty():
            (state, path, priority) = Heap.pop()
            ctr += 1
            if maze.isGoalState(state):
                #print("Reach the goal!")
                return path
            for successor in maze.getSuccessor(state):
                (neighbor, action, cost) = successor
                if neighbor not in visited.keys():
                    visited[neighbor[0]] = True
                    Heap.update((neighbor, path + [action], priority + cost), priority + cost + heuristic(neighbor, maze.goal))
            if state not in visited.keys():
                visited[state] = True
            if Heap.isEmpty(): 
                #print("There is no such a path!")
                return path
    
 
def approximateDistance(maze):
    return len(BFS(maze))

def Astar_Thinning(maze):
    Heap = structure.PriorityQueue()
    Heap.heap = [(0, 1, (maze.start, [], approximateDistance(maze)))]
    simple = maze.simplify(0.6)
    simple.printMaze(Astar, manhattanDistance)
    visited = {}
    while not Heap.isEmpty():
        (state, path, priority) = Heap.pop()
        if maze.isGoalState(state):
            #print("Reach the goal!")
            return path
        simple.start = state
        for successor in maze.getSuccessor(state):
            (neighbor, action, cost) = successor
            if neighbor not in visited.keys():
                visited[neighbor[0]] = True
                Heap.update((neighbor, path + [action], priority + cost), priority + cost + approximateDistance(simple))
        if state not in visited.keys():
            visited[state] = True
        if Heap.isEmpty(): 
            #print("There is no such a path!")
            return path    
        
def expectValue(fireMaze, state, visited):
    '''
        Input: 
            fireMaze: a fireMaze object
            state: current state, determinate the probability of being fire in
                   the future (utility)
            visited: a list of the nodes(states) which have already been expand
        
        check the (current) state's utility by calculating its probability of being 
        fire in the future. Calculation is based on the utility of its neighborhoods
        which means the probability of being fire is cumulative from the nearest fire spot
        
        Output:
            the utility of current state
    '''
    if fireMaze.isGoalState(state):
        fireMaze.utility[state] = (fireMaze.length + fireMaze.width) ** 2
        return (fireMaze.length + fireMaze.width) ** 2
    elif fireMaze.probabilityDistribution[state] == 1:
        fireMaze.utility[state] = -10
        return -10
    value = 0
    for successor in fireMaze.getSuccessor(state):
        (neighbor, _, _) = successor
        if neighbor not in visited:
            probability = fireMaze.probabilityDistribution[neighbor]
            if neighbor not in fireMaze.utility.keys():
                fireMaze.utility[neighbor] = expectValue(fireMaze, neighbor, visited)
            value += probability * fireMaze.utility[neighbor]
    fireMaze.utility[state] = value
    return value 

def FireBFS(fireMaze, state):
        queue = structure.Queue()
        queue.enqueue((state, []))
        visited = {}
        states = [state]
        while not queue.isEmpty():
            (state, path) = queue.dequeue()
            for successor in fireMaze.getSuccessor(state):
                (neighbor, action, _) = successor
                if fireMaze.isGoalState(neighbor) or neighbor in fireMaze.fireSpots:
                    #print("Reach the goal!")
                    path.append(action)
                    utility = 1
                    for action in path:
                        (x, y) = states[-1]
                        if action == "west":
                            states.append((x - 1, y))
                            utility *= fireMaze.probabilityDistribution[(x - 1, y)]
                        elif action == "north":
                            states.append((x, y - 1))
                            utility *= fireMaze.probabilityDistribution[(x, y - 1)]
                        elif action == "east":
                            states.append((x + 1, y))
                            utility *= fireMaze.probabilityDistribution[(x + 1, y)]
                        elif action == "south":
                            utility *= fireMaze.probabilityDistribution[(x, y + 1)]
                    if states[-1] in fireMaze.fireSpots:
                        utility *= (-100)
                        print(utility)
                    else:                       
                        utility = (fireMaze.length + fireMaze.width) ** 2
                    return utility
                elif neighbor not in visited.keys():
                    queue.enqueue((neighbor, path + [action]))
            if state not in visited.keys():
                visited[state] = True
            if queue.isEmpty():
                #print("There is no such a path!")
                utility = 1
                for action in path:
                    (x, y) = states[-1]
                    if action == "west":
                        states.append((x - 1, y))
                        utility *= fireMaze.probabilityDistribution[(x - 1, y)]
                    elif action == "north":
                        states.append((x, y - 1))
                        utility *= fireMaze.probabilityDistribution[(x, y - 1)]
                    elif action == "east":
                        states.append((x + 1, y))
                        utility *= fireMaze.probabilityDistribution[(x + 1, y)]
                    elif action == "south":
                        utility *= fireMaze.probabilityDistribution[(x, y + 1)]
                if states[-1] in fireMaze.fireSpots:
                    utility *= (-100)
                else:
                    utility = utility / fireMaze.probabilityDistribution[states[-1]]
                    utility *= (fireMaze.length + fireMaze.width) ** 2
                return utility
            
def makeDecision(fireMaze):
    visited = [fireMaze.start]
    path = []
    while True:
        state = visited[-1]
        v = -2 ** 64
        print('number of fire spots:', len(fireMaze.fireSpots))
        print('---------')
        print(fireMaze.fireSpots)
        print('---------')
        print('current state is', state, 'the probability of being fire is', fireMaze.probabilityDistribution[state])
        if fireMaze.isGoalState(state):
            print("You escape the fire maze!")
            return path
        elif state in fireMaze.fireSpots:
            print("You are burned")
            return path
        optimalAction = ''
        nextState = state
        for successor in fireMaze.getSuccessor(state):
            (neighbor, action, _) = successor
            if neighbor not in visited and v < FireBFS(fireMaze, neighbor):
                v = FireBFS(fireMaze, neighbor)
                nextState = neighbor
                optimalAction = action
        print(v)
        visited.append(nextState)
        path.append(optimalAction)
        fireMaze.spreadFire()
        fireMaze.distributeFire()
                
        
    
def maxValue(fireMaze):
    '''
        go through the whole maze, when moving to a state, the maze will update
        its fire probability distribution, and based on the new distribution
        function will choose the optimal direction
    '''
    visited = [fireMaze.start]
    path = []
    while visited != []:
        fireMaze.utility = {}
        state = visited[-1]
        nextState = state
        optimalAction = ''
        if fireMaze.isGoalState(state):
            return path
        value = -2 ** 64
        print(1)
        for successor in fireMaze.getSuccessor(state):
            (neighbor, action, _) = successor
            print('state ', neighbor, 'with utility', expectValue(fireMaze, neighbor, visited))
            if neighbor not in visited and value < expectValue(fireMaze, neighbor, visited):
                value = expectValue(fireMaze, neighbor, visited)
                nextState = neighbor
                optimalAction = action
            print(fireMaze.utility)
            return path
        if nextState != state:
            print('current state is ', state)
            print('next state will be', nextState, ', and the action will be', optimalAction)
            visited.append(nextState)
        visited = visited[1:]
        if optimalAction != '':
            path.append(optimalAction)
        fireMaze.spreadFire()
        fireMaze.distributeFire()
        return path
    return path
        
    
        

maze = Maze(10, 10, 0.2)
maze.generateMaze()
path = []
ctr = 0
timeA = datetime.now()
maze.printMaze(BFS)
print("operations of BFS is:", ctr)
timeB = datetime.now()
maze.printMaze(DFS)
print("operations of DFS is:", ctr)
timeC = datetime.now()
maze.printMaze(Astar, manhattanDistance)
print("operations of Astar-manhattan is:", ctr)
timeD = datetime.now()
maze.printMaze(Astar, euclideanDistance)
print("operations of Astar-Euclidean is:", ctr)
timeE = datetime.now()
maze.printMaze(Astar_Thinning)
timeF = datetime.now()

print("Running time of BFS:",timeB-timeA)
print("Running time of DFS:",timeC-timeB)
print("Running time of Astar-Manhattan:",timeD-timeC)
print("Running time of Astar-Euclidean:",timeE-timeD)
print("Running time of Astar-Thinning:",timeF-timeE)

firemaze = FireMaze(10, 10, 0, 1)
firemaze.setFire()
firemaze.distributeFire()
print(makeDecision(firemaze))