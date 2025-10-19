#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 16:41:28 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def GD(x0, y0, lr, epoch):
    f = lambda x, y: x ** 2 - y ** 2
    g_x = lambda x: 2 * x
    x, y = x0, y0
    x_list, y_list, z_list = [], [], []
    for i in range(epoch):
        x_list.append(x)
        y_list.append(y)
        z_list.append(f(x, y) * 1.01)

        grad_x, grad_y = g_x(x), g_x(y)
        x -= lr * grad_x
        y -= lr * grad_y
        print("Epoch{}: grad={} {}, x={}".format(i, grad_x, grad_y, x))
        if abs(grad_x) < 1e-6 and abs(grad_y) < 1e-6:
            break
    return x_list, y_list, z_list


def update(num, x, y, z, ax):
    x, y, z = x[:num], y[:num], z[:num]
    ax.scatter3D(x, y, z, color='black', s=100)
    return ax


def draw_gd():
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.linspace(-3, 3, 1000), np.linspace(-3, 3, 1000))
    z = x ** 2 - y ** 2
    # ax3d = plt.gca(projection='3d')
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    plt.tick_params(labelsize=10)
    ax3d.plot_surface(x, y, z, cstride=20, rstride=20, cmap="jet")

    x_list, y_list, z_list = GD(-3, 0, 0.01, 100)
    x_list, y_list, z_list = np.array(x_list), np.array(y_list), np.array(z_list)

    ani = animation.FuncAnimation(fig, update, frames=25, fargs=(x_list, y_list, z_list, ax3d), interval=50, blit=False)
    ani.save('3D.gif')


if __name__ == '__main__':
    draw_gd()
