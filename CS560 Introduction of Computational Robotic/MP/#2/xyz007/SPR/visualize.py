import sys

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

'''
Set up matplotlib to create a plot with an empty square
'''
def setupPlot():
    fig = plt.figure(num=None, dpi=120, facecolor='w', edgecolor='k')
    ax = fig.subplots()
    ax.set_axisbelow(True)
    ax.grid(which='minor', linestyle=':', alpha=0.2)
    ax.grid(which='major', linestyle=':', alpha=0.5)
    return fig, ax

'''
Make a patch for a single pology 
'''
def createPolygonPatch(polygon):
    verts = []
    codes= []
    for v in range(0, len(polygon)):
        xy = polygon[v]
        verts.append((xy[0], xy[1]))
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='gray', lw=1)

    return patch
    
'''
Make a patch for the robot
'''
def createPolygonPatchForRobot(polygon):
    verts = []
    codes= []
    for v in range(0, len(polygon)):
        xy = polygon[v]
        verts.append((xy[0], xy[1]))
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='gray', lw=1)

    return patch
    

'''
Render polygon obstacles  
'''
def drawPolygons(polygons, fig, ax):
    for p in range(0, len(polygons)):
        patch = createPolygonPatch(polygons[p])
        ax.add_patch(patch)    



if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) < 2):
        print("Please provide input tfile: python visualize.py [env-file]")
        exit()
    
    filename = sys.argv[1]

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

    # Setup
    fig, ax = setupPlot()

    # Draw the polygons
    drawPolygons(polygons, fig, ax)

    # Extra visualization elements goes here
    # ===== delete the following line before you make some changes =====
    ax.plot()
    # ======= delete the above line before you make some changes =======
    
    plt.show()