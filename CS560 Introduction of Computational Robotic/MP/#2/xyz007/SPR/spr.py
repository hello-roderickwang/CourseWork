import sys
import numpy as np
from collections import defaultdict

'''
Report reflexive vertices
'''
def findReflexiveVertices(polygons):
    vertices=[]
    
    # Your code goes here
    # You should return a list of (x,y) values as lists, i.e.
    # vertices = [[x1,y1],[x2,y2],...]
    
    return vertices

'''
Compute the roadmap graph
'''
def computeSPRoadmap(polygons, reflexVertices):
    vertexMap = dict()
    adjacencyListMap = defaultdict(list)
    
    # Your code goes here
    # You should check for each pair of vertices whether the
    # edge between them should belong to the shortest path
    # roadmap. 
    #
    # Your vertexMap should look like
    # {1: [5.2,6.7], 2: [9.2,2.3], ... }
    #
    # and your adjacencyListMap should look like
    # {1: [[2, 5.95], [3, 4.72]], 2: [[1, 5.95], [5,3.52]], ... }
    #
    # The vertex labels used here should start from 1
    
    return vertexMap, adjacencyListMap

'''
Perform uniform cost search 
'''
def uniformCostSearch(adjListMap, start, goal):
    path = []
    pathLength = 0
    
    # Your code goes here. As the result, the function should
    # return a list of vertex labels, e.g.
    #
    # path = [23, 15, 9, ..., 37]
    #
    # in which 23 would be the label for the start and 37 the
    # label for the goal.

    return path, pathLength

'''
Agument roadmap to include start and goal
'''
def updateRoadmap(polygons, vertexMap, adjListMap, x1, y1, x2, y2):
    updatedALMap = dict()
    startLabel = 0
    goalLabel = -1

    # Your code goes here. Note that for convenience, we 
    # let start and goal have vertex labels 0 and -1,
    # respectively. Make sure you use these as your labels
    # for the start and goal vertices in the shortest path
    # roadmap. Note that what you do here is similar to
    # when you construct the roadmap. 

    return startLabel, goalLabel, updatedALMap


if __name__ == "__main__":
    # Retrive file name for input data
    if(len(sys.argv) < 6):
        print("Five arguments required: python spr.py [env-file] [x1] [y1] [x2] [y2]")
        exit()
    
    filename = sys.argv[1]
    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])

    # Read data and parse polygons
    lines = [line.rstrip('\n') for line in open(filename)]
    polygons = []
    for line in range(0, len(lines)):
        xys = lines[line].split(';')
        polygon = []
        for p in range(0, len(xys)):
            polygon.append([float(i) for i in xys[p].split(',')])
        polygons.append(polygon)

    # Print out the data
    print("Pologonal obstacles:")
    for p in range(0, len(polygons)):
        print(str(polygons[p]))
    print("")

    # Compute reflex vertices
    reflexVertices = findReflexiveVertices(polygons)
    print("Reflexive vertices:")
    print(str(reflexVertices))
    print("")

    # Compute the roadmap 
    vertexMap, adjListMap = computeSPRoadmap(polygons, reflexVertices)
    print("Vertex map:")
    print(str(vertexMap))
    print("")
    print("Base roadmap:")
    print(dict(adjListMap))
    print("")

    # Update roadmap
    start, goal, updatedALMap = updateRoadmap(polygons, vertexMap, adjListMap, x1, y1, x2, y2)
    print("Updated roadmap:")
    print(dict(updatedALMap))
    print("")

    # Search for a solution     
    path, length = uniformCostSearch(updatedALMap, start, goal)
    print("Final path:")
    print(str(path))
    print("Final path length:" + str(length))
