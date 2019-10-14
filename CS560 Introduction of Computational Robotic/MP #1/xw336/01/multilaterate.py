import sys
import numpy as np
from scipy.optimize import minimize

def multilaterate(distances):
    def error(x, c, r):
        return sum([(np.linalg.norm(x - c[i]) - r[i]) ** 2 for i in range(len(c))])
    dist = np.array(distances)[:,3]
    coordinate = np.array([np.array(distances)[:,0], np.array(distances)[:,1],np.array(distances)[:,2]]).T
    print('dist:', dist)
    print('coordinate', coordinate)
    l = len(coordinate)
    S = sum(dist)
    W = [((l - 1) * S) / (S - w) for w in dist]
    x0 = sum([W[i] * coordinate[i] for i in range(l)])
    rst = minimize(error, x0, args=(coordinate, dist), method='Nelder-Mead').x
    return rst

if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) == 1):
        print("Please enter data file name.")
        exit()
    
    filename = sys.argv[1]

    # Read data
    lines = [line.rstrip('\n') for line in open(filename)]
    distances = []
    for line in range(0, len(lines)):
        distances.append(list(map(float, lines[line].split(' '))))

    # Print out the data
    print ("The input four points and distances, in the format of [x, y, z, d], are:")
    for p in range(0, len(distances)):
        print (*distances[p]) 

    # Call the function and compute the location 
    location = multilaterate(distances)
    print 
    print ("The location of the point is: " + str(location))
