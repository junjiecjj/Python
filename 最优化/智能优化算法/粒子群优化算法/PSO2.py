#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:56:21 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODY4ODE5MQ==&mid=2247485118&idx=1&sn=5a5cbaf5e5753688c7adbbb763a208f3&chksm=ce00f80df3e4857dc2463c0cf1edf4d72b36a5eb4c1f8e577961b91abb910fb62281a5ff889d&mpshare=1&scene=1&srcid=0901SbJe8vZSFIiYfOgwVrOK&sharer_shareinfo=ef450cc44d0ff9173ff101c614a18658&sharer_shareinfo_first=ef450cc44d0ff9173ff101c614a18658&exportkey=n_ChQIAhIQXIxkH%2Fxc1Er77hAPSA1XqxKfAgIE97dBBAEAAAAAADOIBtW%2F7SwAAAAOpnltbLcz9gKNyK89dVj03b0LY0kUWLSZa4hY9cna3b03qx%2FluPTegQeAXmsIFCE1KPcY6dUT5EweTRsg8oZk0abdwNA0Oo8CFxBZc15fhSYYXP%2FyGSSRlKOiT2gGrzabO%2B9MD4ALQz1lGYdeWwQ0FWNJIFSPUMK8Xs5le%2FwsZhVR2xdCGSRsuUFw8URkKF%2BUohk1IdNrWNcGADqAmC4vDSD9nnNs6NuQgxHqThS%2Fk7U8mmbcZjwPVv6cVhk90k0xpMtVJX6C2hKMNjFlOg%2BNfQpXd5bkTdwQ7m9ZEPXKn17KpkPSexV6zi3D9wP6kJy0QE4b6W8q6f%2FM9itz%2F4mh26TiebAiPT1J&acctmode=0&pass_ticket=fYsPZPqWAUQ9LRFF8OFaY6QKIfs95IR2SrcxPyry3uwUaMc4T8CbvD%2Fa%2FvilK%2Bk2&wx_header=0#rd

"""


import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

x = np.arange(-10,10, 0.01)
y = x**2
plt.plot(x,y, lw=3)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


def function(x):
    y = []
    for i in x:
        if i > -3:
            y.append(i**2-4*i+i**3)
        else:
            y.append(0.2*i**2)
    return y
x = np.arange(-10,3, 0.001)
y = function(x)
plt.plot(x,y, lw=3)
plt.title('优化函数')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()



POP_SIZE = 10 #population size
MAX_ITER = 30 #the amount of optimization iterations
w = 0.2 #inertia weight
c1 = 1 #personal acceleration factor
c2 = 2 #social acceleration factor




def populate(size):
    x1,x2 = -10, 3 #x1, x2 = right and left boundaries of our X axis
    pop = rnd.uniform(x1,x2, size) # size = amount of particles in population
    return pop



x1=populate(50)
y1=function(x1)

plt.plot(x,y, lw=3, label='Func to optimize')
plt.plot(x1,y1,marker='o', ls='', label='Particles')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()




"""Particle Swarm Optimization (PSO)"""
particles = populate(POP_SIZE) #generating a set of particles
velocities = np.zeros(np.shape(particles)) #velocities of the particles
gains = -np.array(function(particles)) #calculating function values for the population

best_positions = np.copy(particles) #it's our first iteration, so all positions are the best
swarm_best_position = particles[np.argmax(gains)] #x with with the highest gain
swarm_best_gain = np.max(gains) #highest gain

l = np.empty((MAX_ITER, POP_SIZE)) #array to collect all pops to visualize afterwards

for i in range(MAX_ITER):

    l[i] = np.array(np.copy(particles)) #collecting a pop to visualize

    r1 = rnd.uniform(0, 1, POP_SIZE) #defining a random coefficient for personal behavior
    r2 = rnd.uniform(0, 1, POP_SIZE) #defining a random coefficient for social behavior

    velocities = np.array(w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)) #calculating velocities

    particles+=velocities #updating position by adding the velocity

    new_gains = -np.array(function(particles)) #calculating new gains

    idx = np.where(new_gains > gains) #getting index of Xs, which have a greater gain now
    best_positions[idx] = particles[idx] #updating the best positions with the new particles
    gains[idx] = new_gains[idx] #updating gains

    if np.max(new_gains) > swarm_best_gain: #if current maxima is greateer than across all previous iters, than assign
        swarm_best_position = particles[np.argmax(new_gains)] #assigning the best candidate solution
        swarm_best_gain = np.max(new_gains) #assigning the best gain

    print(f'Iteration {i+1} \tGain: {swarm_best_gain}')






































































































