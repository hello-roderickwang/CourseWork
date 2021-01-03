import sys
import numpy as np
import itertools

from collections import defaultdict
from typing import Dict, Optional, List, Tuple
from heapq import *

def angle(x: np.array, y: np.array) -> float:
    return np.arccos(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))

def is_inner(x: np.array, y: np.array) -> bool:
    return np.cross(x, y) < 0

def is_reflexive(p1: np.array, p2: np.array, p3: np.array) -> bool:
    x = p2 - p1
    y = p3 - p2
    theta = angle(x, y)
    if is_inner(x, y):
        theta = 2 * np.pi - theta
    return theta > np.pi

def findReflexiveVertices(
        polygons: List[List[List[float]]]) -> List[List[float]]:
    vertices = []
    for polygon in polygons:
        for i in range(len(polygon)):
            p1 = np.array(polygon[i - 1])
            p2 = np.array(polygon[i])
            p3 = np.array(polygon[0] if i == (
                len(polygon) - 1) else polygon[i + 1])
            if is_reflexive(p1, p2, p3):
                vertices.append(polygon[i])
    return vertices

def perp(a: np.array) -> np.array:
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def intersect(
        x1: np.array,
        x2: np.array,
        y1: np.array,
        y2: np.array) -> np.array:
    dx = x2 - x1
    dy = y2 - y1
    dp = x1 - y1
    dxp = perp(dx)
    num = np.dot(dxp, dp)
    denom = np.dot(dxp, dy)
    if denom == 0:
        return np.array([])
    return (num / denom.astype(float)) * dy + y1

def in_same_polygon(polygons: List[List[List[float]]],
                    target: List[float],
                    possible_neighbor: List[float]) -> List[List[float]]:
    for polygon in polygons:
        if target in polygon:
            if possible_neighbor in polygon:
                return polygon
            else:
                return None
    raise ValueError("Parameter \"target\" not in any polygon.")

def is_within_bounds(
        x1: List[float],
        x2: List[float],
        y1: List[float],
        y2: List[float],
        int_pt: List[float]):
    ix, iy = int_pt
    if not min(x1[0], x2[0]) <= ix <= max(x1[0], x2[0]):
        return False
    elif not min(x1[1], x2[1]) <= iy <= max(x1[1], x2[1]):
        return False
    elif not min(y1[0], y2[0]) <= ix <= max(y1[0], y2[0]):
        return False
    elif not min(y1[1], y2[1]) <= iy <= max(y1[1], y2[1]):
        return False
    else:
        return True

def visible_in_polygon(polygon: List[List[float]],
                       target: List[float],
                       possible_neighbor: List[float],
                       clockwise: bool = True) -> bool:
    n = len(polygon)
    i = polygon.index(target)
    result = None
    for num_inc in range(1, n):
        if clockwise:
            j = i + num_inc
        else:
            j = i - num_inc
            if j < 0:
                j = n - j
        j %= n
        if is_reflexive(np.array(polygon[j - 1]),
                        np.array(polygon[j]),
                        np.array(polygon[(j + 1) % n])):
            result = polygon[j]
            break
    return result == possible_neighbor

def visible_outside_polygon(
        polygons: List[List[List[float]]], target: List[float], possible_neighbor: List[float]):
    for polygon in polygons:
        n = len(polygon)
        for i in range(n):
            p_i = polygon[i]
            p_j = polygon[(i + 1) % n]
            if p_i == target or p_i == possible_neighbor:
                continue
            if p_j == target or p_j == possible_neighbor:
                continue
            int_pt = list(
                intersect(
                    np.array(p_i),
                    np.array(p_j),
                    np.array(target),
                    np.array(possible_neighbor)))
            if int_pt and is_within_bounds(
                    target, possible_neighbor, p_i, p_j, int_pt):
                return False
    return True

def computeSPRoadmap(polygons: List[List[List[float]]],
                     reflexVertices: List[List[float]]) -> Tuple[Dict[int,
                                                           List[float]],
                                                           Dict[int,
                                                           List[List[float]]]]:
    vertexMap = dict(enumerate(reflexVertices, 1))
    adjacencyListMap = defaultdict(list)
    for ((target_index, target), (possible_neighbor_index, possible_neighbor)
         ) in itertools.combinations(vertexMap.items(), 2):
        polygon = in_same_polygon(polygons, target, possible_neighbor)
        if polygon:
            if visible_in_polygon(
                    polygon,
                    target,
                    possible_neighbor) or visible_in_polygon(
                    polygon,
                    target,
                    possible_neighbor,
                    clockwise=False):
                target_vec = np.array(target)
                neighbor_vec = np.array(possible_neighbor)
                adjacencyListMap[target_index].append(
                    [possible_neighbor_index, np.linalg.norm(target_vec - neighbor_vec)])
                adjacencyListMap[possible_neighbor_index].append(
                    [target_index, np.linalg.norm(target_vec - neighbor_vec)])
        elif visible_outside_polygon(polygons, target, possible_neighbor):
            target_vec = np.array(target)
            neighbor_vec = np.array(possible_neighbor)
            adjacencyListMap[target_index].append(
                [possible_neighbor_index, np.linalg.norm(target_vec - neighbor_vec)])
            adjacencyListMap[possible_neighbor_index].append(
                [target_index, np.linalg.norm(target_vec - neighbor_vec)])
    return vertexMap, adjacencyListMap

def uniformCostSearch(adjListMap, start, goal):
    queue = [(0, start, ())]
    seen = set()
    mins = {start: 0}
    while queue:
        (cost, v1, path) = heappop(queue)
        if v1 not in seen:
            seen.add(v1)
            path += (v1,)
            if v1 == goal:
                return (list(path), cost)
            for v2, c in adjListMap.get(v1, ()):
                if v2 in seen:
                    continue
                prev = mins.get(v2, None)
                nxt = cost + c
                if not prev or nxt < prev:
                    mins[v2] = nxt
                    heappush(queue, (nxt, v2, path))
    return [], float("inf")

def updateRoadmap(polygons, vertexMap, adjListMap, x1, y1, x2, y2):
    updatedALMap = defaultdict(list)
    startLabel = 0
    goalLabel = -1
    updatedALMap.update(adjListMap)
    for (possible_neighbor_index, possible_neighbor) in vertexMap.items():
        if visible_outside_polygon(polygons, [x1, y1], possible_neighbor):
            dist = np.linalg.norm(
                np.array([x1, y1]) - np.array(possible_neighbor))
            updatedALMap[possible_neighbor_index].append([startLabel, dist])
            updatedALMap[startLabel].append([possible_neighbor_index, dist])
        if visible_outside_polygon(polygons, [x2, y2], possible_neighbor):
            dist = np.linalg.norm(
                np.array([x2, y2]) - np.array(possible_neighbor))
            updatedALMap[possible_neighbor_index].append([goalLabel, dist])
            updatedALMap[goalLabel].append([possible_neighbor_index, dist])
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
    start, goal, updatedALMap = updateRoadmap(
        polygons, vertexMap, adjListMap, x1, y1, x2, y2)
    print("Updated roadmap:")
    print(dict(updatedALMap))
    print("")

    # Search for a solution
    path, length = uniformCostSearch(updatedALMap, start, goal)
    print("Final path:")
    print(str(path))
    print("Final path length:" + str(length))