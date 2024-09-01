

# https://mp.weixin.qq.com/s?__biz=Mzg3ODY4ODE5MQ==&mid=2247486091&idx=2&sn=e47a44b9793f2190d8376628017c7ae0&chksm=ce406577f409b3b74198dcf5bfc5dd9e74c742826f433262ccc2b9548311c317b8640b90dbc3&mpshare=1&scene=1&srcid=0901ZKQkMtdPf0o8EwOFrUu8&sharer_shareinfo=86c53c18f0bc074b492819b510e7a96e&sharer_shareinfo_first=86c53c18f0bc074b492819b510e7a96e&exportkey=n_ChQIAhIQpuXnqPPI8AW46%2BjEpIqCHxKKAgIE97dBBAEAAAAAAJBKKsdDzjkAAAAOpnltbLcz9gKNyK89dVj0JUwrtvX3z44%2FYlLz4TK%2BJEi6xDqZzRXBkqGHc7dZ7jvSXVSfM%2FG3e1fgrvAgLKe%2B7RI4oEKYi0A6WdE3T7j6NgtddswFawKjzWjT5gcDGG7A2093o5ZEwgQuLZtfhQNVVf%2ByTIYLOrevjnFhISGVnMs%2FTSkiTFZuLlNj8ZbT6Qf1hGHy63FEp7%2FJVkJ8Ian72XFSIodK04ju22XjeaX%2B8lu32vX%2B0iYfGBGZuAvVhDHNlXCCqar2zOG1n4Ofd%2F5p4DvnmT4CylBJ0Gf3GtxLxzhS%2F8o6CFKDmerHCoy1W7iVbmEx&acctmode=0&pass_ticket=GfbcPlVA7itNCKYhnFD1Gr67mwM4lu95IDEvaqQDU2zZgXplkskUkvNFFT8%2FWgzS&wx_header=0#rd

import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


# 2.定义适应度函数
# Fitness function
# We assume the problem can be expressed by the following equation:
# f(x1,x2)=(x1+2*-x2+3)^2 + (2*x1+x2-8)^2
# The objective is to find a minimum which is 0

def fitness_function(x1,x2):
  f1=x1+2*-x2+3
  f2=2*x1+x2-8
  z = f1**2+f2**2
  return z

# 3. 更新速度
def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, max=1.0, c=0.1):
  # Initialise new velocity array
  num_particle = len(particle)
  new_velocity = np.array([0.0 for i in range(num_particle)])
  # Randomly generate r1, r2 and inertia weight from normal distribution
  r1 = random.uniform(0,max)
  r2 = random.uniform(0,max)
  w = random.uniform(w_min,max)
  c1 = c
  c2 = c
  # Calculate new velocity
  for i in range(num_particle):
    new_velocity[i] = w*velocity[i] + c1*r1*(pbest[i]-particle[i])+c2*r2*(gbest[i]-particle[i])
  return new_velocity

# 4. 更新位置
def update_position(particle, velocity):
  # Move particles by adding velocity
  new_particle = particle + velocity
  return new_particle


# 5. PSO的主要功能
def pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion):
  # Initialisation
  # Population
  particles = [[random.uniform(position_min, position_max) for j in range(dimension)] for i in range(population)]
  # Particle's best position
  pbest_position = particles
  # Fitness
  pbest_fitness = [fitness_function(p[0],p[1]) for p in particles]
  # Index of the best particle
  gbest_index = np.argmin(pbest_fitness)
  # Global best particle position
  gbest_position = pbest_position[gbest_index]
  # Velocity (starting from 0 speed)
  velocity = [[0.0 for j in range(dimension)] for i in range(population)]

  # Loop for the number of generation
  for t in range(generation):
    # Stop if the average fitness value reached a predefined success criterion
    if np.average(pbest_fitness) <= fitness_criterion:
      break
    else:
      for n in range(population):
        # Update the velocity of each particle
        velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)
        # Move the particles to new position
        particles[n] = update_position(particles[n], velocity[n])
    # Calculate the fitness value
    pbest_fitness = [fitness_function(p[0],p[1]) for p in particles]
    # Find the index of the best particle
    gbest_index = np.argmin(pbest_fitness)
    # Update the position of the best particle
    gbest_position = pbest_position[gbest_index]

  # Print the results
  print('Global Best Position: ', gbest_position)
  print('Best Fitness Value: ', min(pbest_fitness))
  print('Average Particle Best Fitness Value: ', np.average(pbest_fitness))
  print('Number of Generation: ', t)
  return particles

# 6. 设置参数值并运行算法
population = 100
dimension = 2
position_min = -100.0
position_max = 100.0
generation = 400
fitness_criterion = 10e-4

# Plotting prepartion
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
x = np.linspace(position_min, position_max, 80)
y = np.linspace(position_min, position_max, 80)
X, Y = np.meshgrid(x, y)
Z= fitness_function(X,Y)
ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.2)

particles = pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion)

# Animation image placeholder
images = []

# Add plot for each generation (within the generation for-loop)
image = ax.scatter3D([ particles[n][0] for n in range(population)], [particles[n][1] for n in range(population)], [fitness_function(particles[n][0],particles[n][1]) for n in range(population)], c='b')

images.append([image])

# Generate the animation image and save
animated_image = animation.ArtistAnimation(fig, images)
# animated_image.save('./pso_simple.gif', writer='pillow')



























































































