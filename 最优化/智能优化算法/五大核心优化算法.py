


# https://mp.weixin.qq.com/s?__biz=Mzg3ODY4ODE5MQ==&mid=2247487343&idx=1&sn=51f0104c478d47ca2bbd8533816256b9&chksm=ceb09c46a451f2f760763b77b4683febae6b63de3f809e091db1d0b563fa7869ea0aef4aa432&mpshare=1&scene=1&srcid=09017RS9UISS899VwsP2BCiN&sharer_shareinfo=0d44c7465cdaf6ce52f39691997465e4&sharer_shareinfo_first=0d44c7465cdaf6ce52f39691997465e4&exportkey=n_ChQIAhIQ89SB054iH%2BHrF81%2B5tPOqRKfAgIE97dBBAEAAAAAAJNpE%2BoqZQwAAAAOpnltbLcz9gKNyK89dVj0yDBObeQnCWHgorKJ5mBdhQBQ47beC9UkJXTBOGlTE1%2BBOUvE%2FOkYNpJSsL6bZsUpmpRWdrOtcU%2FzvsIo4DRkQORbX6FnOY7MPqvA%2B%2BOGJ%2BcUYm2HOBEPCepz8Sj%2FGnloR96ppV9h2AUZIrEZZGNcsRtdNFM2Q0Y5mmMDRiTdofBbZzx%2FmIGA79n09hX0nLFv16YuXZps1jMI8YRROkOZogISmiza3auJP9DxrtyM3u%2Bb1RQxS7ReJG9%2F7AjarBzq5a7oQomIk%2FQIoE3HcYv3PdU%2BksCnGrz%2BTQZSrO%2BO93rdYYV2u43A3HVVKl5zR%2Ft5j8L7NVk2sOWH&acctmode=0&pass_ticket=1GNYoSMsVa%2FVrZYcKM4gSYKRbXnZZWlqHMA0nUHVzqIXBUWPVly%2BOWvAmTKJl5b1&wx_header=0#rd





#%%>>>>>>>>>>>>>>>>>>>>>>>>>  遗传算法 (Genetic Algorithm):

# Python3 program to create target string, starting from
# random string using Genetic Algorithm

import random

# Number of individuals in each generation
POPULATION_SIZE = 100

# Valid genes
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

# Target string to be generated
TARGET = "I love GeeksforGeeks"

class Individual(object):
  '''
  Class representing individual in population
  '''
  def __init__(self, chromosome):
    self.chromosome = chromosome
    self.fitness = self.cal_fitness()

  @classmethod
  def mutated_genes(self):
    '''
    create random genes for mutation
    '''
    global GENES
    gene = random.choice(GENES)
    return gene

  @classmethod
  def create_gnome(self):
    '''
    create chromosome or string of genes
    '''
    global TARGET
    gnome_len = len(TARGET)
    return [self.mutated_genes() for _ in range(gnome_len)]

  def mate(self, par2):
    '''
    Perform mating and produce new offspring
    '''

    # chromosome for offspring
    child_chromosome = []
    for gp1, gp2 in zip(self.chromosome, par2.chromosome):

      # random probability
      prob = random.random()

      # if prob is less than 0.45, insert gene
      # from parent 1
      if prob < 0.45:
        child_chromosome.append(gp1)

      # if prob is between 0.45 and 0.90, insert
      # gene from parent 2
      elif prob < 0.90:
        child_chromosome.append(gp2)

      # otherwise insert random gene(mutate),
      # for maintaining diversity
      else:
        child_chromosome.append(self.mutated_genes())

    # create new Individual(offspring) using
    # generated chromosome for offspring
    return Individual(child_chromosome)

  def cal_fitness(self):
    '''
    Calculate fitness score, it is the number of
    characters in string which differ from target
    string.
    '''
    global TARGET
    fitness = 0
    for gs, gt in zip(self.chromosome, TARGET):
      if gs != gt: fitness+= 1
    return fitness

# Driver code
def main():
  global POPULATION_SIZE

  #current generation
  generation = 1

  found = False
  population = []

  # create initial population
  for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))

  while not found:

    # sort the population in increasing order of fitness score
    population = sorted(population, key = lambda x:x.fitness)

    # if the individual having lowest fitness score ie.
    # 0 then we know that we have reached to the target
    # and break the loop
    if population[0].fitness <= 0:
      found = True
      break

    # Otherwise generate new offsprings for new generation
    new_generation = []

    # Perform Elitism, that mean 10% of fittest population
    # goes to the next generation
    s = int((10*POPULATION_SIZE)/100)
    new_generation.extend(population[:s])

    # From 50% of fittest population, Individuals
    # will mate to produce offspring
    s = int((90*POPULATION_SIZE)/100)
    for _ in range(s):
      parent1 = random.choice(population[:50])
      parent2 = random.choice(population[:50])
      child = parent1.mate(parent2)
      new_generation.append(child)

    population = new_generation

    print("Generation: {}\tString: {}\tFitness: {}".\
      format(generation,
      "".join(population[0].chromosome),
      population[0].fitness))

    generation += 1


  print("Generation: {}\tString: {}\tFitness: {}".\
    format(generation,
    "".join(population[0].chromosome),
    population[0].fitness))

if __name__ == '__main__':
  main()


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 粒子群优化 (Particle Swarm Optimization, PSO):

# python implementation of particle swarm optimization (PSO)
# minimizing rastrigin and sphere function

import random
import math # cos() for Rastrigin
import copy # array-copying convenience
import sys   # max float


#-------fitness functions---------

# rastrigin function
def fitness_rastrigin(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitnessVal

#sphere function
def fitness_sphere(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi*xi)
    return fitnessVal
#-------------------------

#particle class
class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)

        # initialize position of the particle with 0.0 value
        self.position = [0.0 for i in range(dim)]

        # initialize velocity of the particle with 0.0 value
        self.velocity = [0.0 for i in range(dim)]

        # initialize best particle position of the particle with 0.0 value
        self.best_part_pos = [0.0 for i in range(dim)]

        # loop dim times to calculate random position and velocity
        # range of position and velocity is [minx, max]
        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
            self.velocity[i] = ((maxx - minx) * self.rnd.random() + minx)

        # compute fitness of particle
        self.fitness = fitness(self.position) # curr fitness

        # initialize best position and fitness of this particle
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness # best fitness

# particle swarm optimization function
def pso(fitness, max_iter, n, dim, minx, maxx):
    # hyper parameters
    w = 0.729 # inertia
    c1 = 1.49445 # cognitive (particle)
    c2 = 1.49445 # social (swarm)

    rnd = random.Random(0)

    # create n random particles
    swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]

    # compute the value of best_position and best_fitness in swarm
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessVal = sys.float_info.max # swarm best

    # computer best particle of swarm and it's fitness
    for i in range(n): # check each particle
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position)

    # main loop of pso
    Iter = 0
    while Iter < max_iter:
        # after every 10 iterations
        # print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)
        for i in range(n): # process each particle
            # compute new velocity of curr particle
            for k in range(dim):
                r1 = rnd.random() # randomizations
                r2 = rnd.random()
                swarm[i].velocity[k] = ((w * swarm[i].velocity[k]) + (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) + (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k])) )
            # if velocity[k] is not in [minx, max]
            # then clip it
            if swarm[i].velocity[k] < minx:
                swarm[i].velocity[k] = minx
            elif swarm[i].velocity[k] > maxx:
                swarm[i].velocity[k] = maxx
        # compute new position using new velocity
        for k in range(dim):
            swarm[i].position[k] += swarm[i].velocity[k]
            # compute fitness of new position
            swarm[i].fitness = fitness(swarm[i].position)
        # is new position a new best for the particle?
        if swarm[i].fitness < swarm[i].best_part_fitnessVal:
            swarm[i].best_part_fitnessVal = swarm[i].fitness
            swarm[i].best_part_pos = copy.copy(swarm[i].position)
        # is new position a new best overall?
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position)
        # for-each particle
        Iter += 1
    #end_while
    return best_swarm_pos
    # end pso

#----------------------------
# Driver code for rastrigin function

print("\nBegin particle swarm optimization on rastrigin function\n")
dim = 3
fitness = fitness_rastrigin


print("Goal is to minimize Rastrigin's function in " + str(dim) + " variables")
print("Function has known min = 0.0 at (", end="")
for i in range(dim-1):
    print("0, ", end="")
    print("0)")

num_particles = 50
max_iter = 100

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter = " + str(max_iter))
print("\nStarting PSO algorithm\n")

best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nPSO completed\n")
print("\nBest solution found:")
print(["%.6f"%best_position[k] for k in range(dim)])
fitnessVal = fitness(best_position)
print("fitness of best solution = %.6f" % fitnessVal)

print("\nEnd particle swarm for rastrigin function\n")

# Driver code for Sphere function
print("\nBegin particle swarm optimization on sphere function\n")
dim = 3
fitness = fitness_sphere

print("Goal is to minimize sphere function in " + str(dim) + " variables")
print("Function has known min = 0.0 at (", end="")
for i in range(dim-1):
    print("0, ", end="")
    print("0)")

num_particles = 50
max_iter = 100

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter = " + str(max_iter))
print("\nStarting PSO algorithm\n")

best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nPSO completed\n")
print("\nBest solution found:")
print(["%.6f"%best_position[k] for k in range(dim)])
fitnessVal = fitness(best_position)
print("fitness of best solution = %.6f" % fitnessVal)

print("\nEnd particle swarm for sphere function\n")


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 模拟退火 (Simulated Annealing):

import math
import random

# Objective function: Rastrigin function
def objective_function(x):
    return 10 * len(x) + sum([(xi**2 - 10 * math.cos(2 * math.pi * xi)) for xi in x])

# Neighbor function: small random change
def get_neighbor(x, step_size=0.1):
    neighbor = x[:]
    index = random.randint(0, len(x) - 1)
    neighbor[index] += random.uniform(-step_size, step_size)
    return neighbor

# Simulated Annealing function
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    # Initial solution
    best = [random.uniform(bound[0], bound[1]) for bound in bounds]
    best_eval = objective(best)
    current, current_eval = best, best_eval
    scores = [best_eval]

    for i in range(n_iterations):
        # Decrease temperature
        t = temp / float(i + 1)
        # Generate candidate solution
        candidate = get_neighbor(current, step_size)
        candidate_eval = objective(candidate)
        # Check if we should keep the new solution
        if candidate_eval < best_eval or random.random() < math.exp((current_eval - candidate_eval) / t):
            current, current_eval = candidate, candidate_eval
            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval
                scores.append(best_eval)

        # Optional: print progress
        if i % 100 == 0:
            print(f"Iteration {i}, Temperature {t:.3f}, Best Evaluation {best_eval:.5f}")

    return best, best_eval, scores

# Define problem domain
bounds = [(-5.0, 5.0) for _ in range(2)]  # for a 2-dimensional Rastrigin function
n_iterations = 1000
step_size = 0.1
temp = 10

# Perform the simulated annealing search
best, score, scores = simulated_annealing(objective_function, bounds, n_iterations, step_size, temp)

print(f'Best Solution: {best}')
print(f'Best Score: {score}')


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 蚁群算法 (Ant Colony Optimization, ACO):

#Ant colony optimization to solve TSP problem
import numpy as np
import random as rd

def lengthCal(antPath,distmat):         #Calculate distance
    length =[]
    dis = 0
    for i in range(len(antPath)):
        for j in range(len(antPath[i]) - 1):
            dis += distmat[antPath[i][j]][antPath[i][j + 1]]
        dis += distmat[antPath[i][-1]][antPath[i][0]]
        length.append(dis)
        dis = 0
    return length

distmat = np.array([[0,35,29,67,60,50,66,44,72,41,48,97],
                 [35,0,34,36,28,37,55,49,78,76,70,110],
                 [29,34,0,58,41,63,79,68,103,69,78,130],
                 [67,36,58,0,26,38,61,80,87,110,100,110],
                 [60,28,41,26,0,61,78,73,103,100,96,130],
                 [50,37,63,38,61,0,16,64,50,95,81,95],
                 [66,55,79,61,78,16,0,49,34,82,68,83],
                 [44,49,68,80,73,64,49,0,35,43,30,62],
                 [72,78,103,87,103,50,34,35,0,47,32,48],
                 [41,76,69,110,100,95,82,43,47,0,26,74],
                 [48,70,78,100,96,81,68,30,32,26,0,58],
                 [97,110,130,110,130,95,83,62,48,74,58,0]])

antNum = 12                   #Ant number
alpha = 1                     #Pheromone importance factor
beta = 3                      #Heuristic function importance factor
pheEvaRate = 0.3              #Pheromone evaporation rate
cityNum = distmat.shape[0]
pheromone = np.ones((cityNum,cityNum))                   #Pheromone matrix
heuristic = 1 / (np.eye(cityNum) + distmat) - np.eye(cityNum)       #Heuristic information matrix, take 1/dismat
iter,itermax = 1,100                       #Number of iterations

while iter < itermax:
    antPath = np.zeros((antNum, cityNum)).astype(int) - 1   # Ant's path
    firstCity = [i for i in range(12)]
    rd.shuffle(firstCity)          #Randomly assign a starting city for each ant
    unvisted = []
    p = []
    pAccum = 0
    for i in range(len(antPath)):
        antPath[i][0] = firstCity[i]
    for i in range(len(antPath[0]) - 1):       #Gradually update the next city each ant is going to
        for j in range(len(antPath)):
            for k in range(cityNum):
                if k not in antPath[j]:
                    unvisted.append(k)
            for m in unvisted:
                pAccum += pheromone[antPath[j][i]][m] ** alpha * heuristic[antPath[j][i]][m] ** beta
            for n in unvisted:
                p.append(pheromone[antPath[j][i]][n] ** alpha * heuristic[antPath[j][i]][n] ** beta / pAccum)
            roulette = np.array(p).cumsum()               #Generate Roulette
            r = rd.uniform(min(roulette), max(roulette))
            for x in range(len(roulette)):
                if roulette[x] >= r:                      #Use the roulette method to choose the next city to go
                    antPath[j][i + 1] = unvisted[x]
                    break
            unvisted = []
            p = []
            pAccum = 0
    pheromone = (1 - pheEvaRate) * pheromone            #Pheromone volatile
    length = lengthCal(antPath,distmat)
    for i in range(len(antPath)):
        for j in range(len(antPath[i]) - 1):
            pheromone[antPath[i][j]][antPath[i][j + 1]] += 1 / length[i]     #Pheromone update
        pheromone[antPath[i][-1]][antPath[i][0]] += 1 / length[i]
    iter += 1
print("The shortest distance is:")
print(min(length))
print("The shortest path is:")
print(antPath[length.index(min(length))])


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 梯度下降法 (Gradient Descent):

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# 定义要最小化的函数（简单的二次函数）
def  f ( x, y ):
    return x** 2 + y** 2

# 定义函数关于 x 和 y 的偏导数
def  df_dx ( x, y ):
    return  2 * x

def  df_dy ( x, y ):
    return  2 * y

# 定义梯度下降算法
def  gradient_descent ( start_x, start_y, learning_rate, num_iterations ):
    # 初始化参数
    x = start_x
    y = start_y
    history = []

    # 执行梯度下降迭代
    for i in  range (num_iterations):
        # 计算梯度
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)

        # 更新参数
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

        # 保存参数的历史记录
        history.append((x, y, f(x, y)))

    return x, y, f(x, y), history

# 定义用于绘制函数的网格
x_range = np.arange(- 10 , 10 , 0.1 )
y_range = np.arange(- 10 , 10 , 0.1 )
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# 执行梯度下降并绘制结果
start_x, start_y = 8 , 8
learning_rate = 0.1
num_iterations = 20
x_opt, y_opt, f_opt, history = gradient_descent(start_x, start_y, learning_rate, num_iterations)


fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (12,12))
ax.set_proj_type('ortho')

## 1
norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
# ax.plot_wireframe(X, Y, Z, color = '#0070C0', linewidth = 0.5)
ax.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5)

## 2
# norm_plt = plt.Normalize(Z.min(), Z.max())
# colors = cm.turbo(norm_plt(Z))
# surf = ax.plot_surface(X, Y, Z,
#                        facecolors=colors,
#                        linewidth=1, # 线宽
#                        shade=False) # 删除阴影
# surf.set_facecolor((0,0,0,0))
# ax.set_proj_type('ortho')

ax.scatter(*zip(*history), c = 'r' , marker= 'o', s = 30 )

ax.set_proj_type('ortho')
ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())

# ax.view_init(azim=-135, elev=30)
ax.grid(False)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
# fig.savefig('Figures/只保留网格线.svg', format='svg')
plt.show()






















































































































































































































































































































































































