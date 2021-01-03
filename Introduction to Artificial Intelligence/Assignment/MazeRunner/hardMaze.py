# -*- coding: utf-8 -*-
"""
Copyright <2019> <Xuenan(Roderick) Wang>

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import easyMaze


def random_hard(my_maze):
    my_maze.generateMaze()
    path = easyMaze.DFS(my_maze)
    if easyMaze.havePath == 1:
        return path
        '''
        print("Have a path!")
        print(len(path))
        '''
    else:
        return 0
        '''
        print("No path!")
        print(len(path))
        '''


if __name__ == '__main__':
    ctr = 0
    failure = 0
    success = 0
    last_len = 0
    path_len = []
    while success < 49999:
        myMaze = easyMaze.Maze(10, 10, 0.3)
        path = random_hard(myMaze)
        if path == 0:
            failure += 1
        elif last_len >= len(path):
            success += 1
            path_len.append(len(path))
        else:
            best_maze = myMaze
            last_len = len(path)
            success += 1
            path_len.append(len(path))
        ctr += 1
    print(path_len)
    print("Best path lenth is:",last_len)
    best_maze.printMaze(easyMaze.DFS)


