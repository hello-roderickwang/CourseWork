#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-10-05 23:23:24
# @Author  : Xuenan(Roderick) Wang
# @Email   : roderick_wang@outlook.com
# @Github  : https://github.com/hello-roderickwang

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def phi(x):
    return 0.5+(np.sign(x)/2)*math.sqrt(1-math.exp(-(2*x*x)/math.pi))

list_y = np.zeros(50)
list_x = np.arange(-5, 5, 0.2)
for i in range(50):
    list_y[i] = phi(list_x[i])

def get_sample(y_array):
    x_array = np.zeros(len(y_array))
    for i in range(len(y_array)):
        for j in range(len(list_y)):
            if y_array[i]<list_y[j]:
                x_array[i] = list_x[j]
                break
    return x_array

if __name__ == '__main__':
    sample50 = np.random.rand(50)
    sample100 = np.random.rand(100)
    sample200 = np.random.rand(200)
    sample500 = np.random.rand(500)
    result50 = np.zeros(50)
    result100 = np.zeros(100)
    result200 = np.zeros(200)
    result500 = np.zeros(500)
    result50 = get_sample(sample50)
    result100 = get_sample(sample100)
    result200 = get_sample(sample200)
    result500 = get_sample(sample500)
    fig, axs = plt.subplots(1, 4, sharey=True, tight_layout=True)
    axs[0].hist(result50, bins=50)
    axs[0].set_xlim(-5, 5, 0.2)
    axs[1].hist(result100, bins=50)
    axs[1].set_xlim(-5, 5, 0.2)
    axs[2].hist(result200, bins=50)
    axs[2].set_xlim(-5, 5, 0.2)
    axs[3].hist(result500, bins=50)
    axs[3].set_xlim(-5, 5, 0.2)
    plt.show()