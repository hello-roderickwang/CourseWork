import sys
import numpy as np
from collections import defaultdict

'''
Report reflexive vertices
'''
def findReflexiveVertices(polygons):
    vertices = []

    # Your code goes here
    # You should return a list of (x,y) values as lists, i.e.
    # vertices = [[x1,y1],[x2,y2],...]

    for n in range(0, len(polygons)):
        for point in range(0, len(polygons[n])):
            point_a = point
            point_b = point+1
            point_c = point+2
            if point_b >= len(polygons[n]):
                point_b = point_b - len(polygons[n])
            if point_c >= len(polygons[n]):
                point_c = point_c - len(polygons[n])
            s = [polygons[n][point_b][0]-polygons[n][point_a][0],\
                 polygons[n][point_b][1]-polygons[n][point_a][1]]
            g = [polygons[n][point_c][0]-polygons[n][point_b][0],\
                 polygons[n][point_c][1]-polygons[n][point_b][1]]
            if s[0]*g[1]-g[0]*s[1] < 0:
                vertices.append([polygons[n][point_b][0], polygons[n][point_b][1]])
    return vertices

'''
Compute the roadmap graph
'''
def computeSPRoadmap(polygons, reflexVertices):
    vertexMap = dict()
    adjacencyListMap = dict()

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

    for n in range(0, len(reflexVertices)):
        vertexMap[n+1] = reflexVertices[n]
    for n in range(1, len(vertexMap)+1):
        list = []
        for m in range(1, len(vertexMap)+1):
            if n == m:
                continue
            k = 0
            for i in range(0, len(polygons)):
                for point in range(0, len(polygons[i])):
                    point_a = point
                    point_b = point+1
                    if point_b >= len(polygons[i]):
                        point_b = point_b - len(polygons[i])
                    s1 = [polygons[i][point_a][0]-vertexMap[n][0],\
                          polygons[i][point_a][1]-vertexMap[n][1]]
                    g1 = [polygons[i][point_b][0]-polygons[i][point_a][0],\
                          polygons[i][point_b][1]-polygons[i][point_a][1]]
                    s2 = [polygons[i][point_a][0]-vertexMap[m][0],\
                          polygons[i][point_a][1]-vertexMap[m][1]]
                    g2 = [polygons[i][point_b][0]-polygons[i][point_a][0],\
                          polygons[i][point_b][1]-polygons[i][point_a][1]]
                    s3 = [vertexMap[m][0]-vertexMap[n][0],\
                          vertexMap[m][1]-vertexMap[n][1]]
                    g3 = [polygons[i][point_a][0]-vertexMap[m][0],\
                          polygons[i][point_a][1]-vertexMap[m][1]]
                    s4 = [vertexMap[m][0]-vertexMap[n][0],\
                          vertexMap[m][1]-vertexMap[n][1]]
                    g4 = [polygons[i][point_b][0]-vertexMap[m][0],\
                          polygons[i][point_b][1]-vertexMap[m][1]]
                    if (s1[0]*g1[1]-g1[0]*s1[1])*(s2[0]*g2[1]-g2[0]*s2[1]) < 0 and\
                       (s3[0]*g3[1]-g3[0]*s3[1])*(s4[0]*g4[1]-g4[0]*s4[1]) < 0:
                       k = k+1
            if k == 0:
                list.append([m, np.around(np.sqrt((vertexMap[n][0]-vertexMap[m][0])**2+\
                (vertexMap[n][1]-vertexMap[m][1])**2), 4)])
        adjacencyListMap[n] = list

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

    distance_from_start = {start: 0}
    previous = {start: start}
    queue = [start]
    while queue:
        temp = min(distance_from_start, key=distance_from_start.get)
        nextp = queue.pop(temp)
        if nextp == goal:
            x = -1
            path.append(x)
            while x != 0:
                path.insert(0, previous[x])
                x = previous[x]
            return path, pathLength

        N = adjListMap[nextp]
        for n in N:
            new_distance = distance_from_start[nextp] + n[1]
            if (n[0] not in previous) or (new_distance < distance_from_start[n[0]]):
                queue.append(n[0])
            if ((n[0] not in distance_from_start)):
                distance_from_start[n[0]] = new_distance
                previous[n[0]] = nextp
                pathLength = new_distance

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

    sg = [[x2, y2], [x1, y1]]
    updatedALMap = adjListMap.copy()
    for m in range(-1, 1):
        list = []
        for n in range(1, len(vertexMap)+1):
            k = 0
            for i in range(0, len(polygons)):
                for point in range(0, len(polygons[i])):
                    point_a = point
                    point_b = point+1
                    if point_b >= len(polygons[i]):
                        point_b = point_b - len(polygons[i])
                    s1 = [polygons[i][point_a][0]-sg[m+1][0],\
                          polygons[i][point_a][1]-sg[m+1][1]]
                    g1 = [polygons[i][point_b][0]-polygons[i][point_a][0],\
                          polygons[i][point_b][1]-polygons[i][point_a][1]]
                    s2 = [polygons[i][point_a][0]-vertexMap[n][0],\
                          polygons[i][point_a][1]-vertexMap[n][1]]
                    g2 = [polygons[i][point_b][0]-polygons[i][point_a][0],\
                          polygons[i][point_b][1]-polygons[i][point_a][1]]
                    s3 = [vertexMap[n][0]-sg[m+1][0],\
                          vertexMap[n][1]-sg[m+1][1]]
                    g3 = [polygons[i][point_a][0]-vertexMap[n][0],\
                          polygons[i][point_a][1]-vertexMap[n][1]]
                    s4 = [vertexMap[n][0]-sg[m+1][0],\
                          vertexMap[n][1]-sg[m+1][1]]
                    g4 = [polygons[i][point_b][0]-vertexMap[n][0],\
                          polygons[i][point_b][1]-vertexMap[n][1]]
                    if (s1[0]*g1[1]-g1[0]*s1[1])*(s2[0]*g2[1]-g2[0]*s2[1]) < 0 and\
                       (s3[0]*g3[1]-g3[0]*s3[1])*(s4[0]*g4[1]-g4[0]*s4[1]) < 0:
                       k = k+1
            if k == 0:
                list.append([n, np.around(np.sqrt((sg[m+1][0]-vertexMap[n][0])**2+\
                (sg[m+1][1]-vertexMap[n][1])**2), 4)])
                updatedALMap[n].append([m, np.around(np.sqrt((sg[m+1][0]-vertexMap[n][0])**2+\
                (sg[m+1][1]-vertexMap[n][1])**2), 4)])
        updatedALMap[m] = list

    return startLabel, goalLabel, updatedALMap

save_vertexMap = 0
save_updateALMap = 0
save_start = 0
save_goal = 0

def save_info(vertexMap, updatedALMap, x1, y1, x2, y2):
    save_vertexMap = vertexMap
    save_updateALMap = updatedALMap
    save_start = [x1, y1]
    save_goal = [x2, y2]

def get_info():
    return save_vertexMap, save_updateALMap, save_start, save_goal

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

    save_info(vertexMap, updatedALMap, x1, y1, x2, y2)
