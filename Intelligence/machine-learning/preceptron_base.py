# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:44:54 2019

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt#绘图工具


def draw(w, b, label='before'):
    fontsize = 15
    plt.figure(1, figsize=(8, 6))
    plt.clf()  # clear current figure
    plt.title(label)
    plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Set1, edgecolors='k')
    plt.xlabel('Sepal length', fontsize=fontsize)
    plt.ylabel('Sepal width', fontsize=fontsize)
    plt.xlim(x_min, x_max)#设定x坐标上下限
    plt.ylim(y_min, y_max)#设定y坐标上下限
    ys = list(map(lambda x: (-w[0] * x + b) / w[1], [x_min, x_max]))
    plt.plot([x_min, x_max], ys)
    plt.show()


if __name__ == '__main__':
    data = np.array([[-1, 1], [1, 3], [0, -2],[2,3],[3,2],[2,4],[1,2]])
    target = [-1, 1, -1,1,-1,-1,1]
    shape = data.shape
    # random initial
    np.random.seed(10)

    w = np.random.random(shape[-1])
    b = np.random.random()

    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    draw(w, b)

    lamb = 0.5
    MAX_TIME = 1000
    for t in range(MAX_TIME):
        change = False
        for i in range(shape[0]):
            if target[i] * (np.dot(w, data[i]) + b) <= 0:
                w += lamb * target[i] * data[i]
                b += lamb * target[i]
                change = True
        if not change:
            break

    draw(w, b, 'after')
