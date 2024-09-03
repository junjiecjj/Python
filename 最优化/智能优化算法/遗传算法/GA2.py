#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:08:52 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODY4ODE5MQ==&mid=2247487199&idx=1&sn=de305ab12fb9e7ebe356b92e8932dcc8&chksm=ce8087c29dddc109ff330a43b17c80239db1e89d7ba8a227bea57fbbdab9805b94a6de4a6073&mpshare=1&scene=1&srcid=0901mMJid74VApCdye7RYiQN&sharer_shareinfo=f6fd2fb46c8e0cf02da2a79913d675f5&sharer_shareinfo_first=f6fd2fb46c8e0cf02da2a79913d675f5&exportkey=n_ChQIAhIQbQQQwa2WEw7cw8OzJoqQrxKfAgIE97dBBAEAAAAAANW0NXAFjMsAAAAOpnltbLcz9gKNyK89dVj0nhRWXKzmRtDxnj5LxQ9JPFS8T%2FR1%2F11EGzSTASs8Tm7NzJshko28MC3oZc4yaR532hbzuXqUdoDONj9t6wkK432m4klfUL23jIxqCfKhfyP2JVMLOOhgytQYxH7Fbye9pP9EMiZOHu%2FXm%2B90VFSKZVCzUt13ityGikNHLXO5YpL%2FRiRRVHFAAbp43GQjJKG26FDkUgMkVrr4occXTbaoet4pS8K89sN9nlBN9Bsnr3Kwvc1Q4kziluqVKClRCSMfU5b2x8LHvjP8dWLdWhudUKvcbC1w7QCVilZr9TV%2F34%2BGTMcSTEfyzK6a4E65uZteUl%2BFyBVVzQAA&acctmode=0&pass_ticket=6rCV5KBOggRuX%2BtDCJRibQR9DCiqBp04OekwoBE%2B4NDYlOfg%2BAtE69s0gtzMXsC2&wx_header=0#rd


"""



import numpy as np
import random
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# Generate random cities
def create_cities(num_cities):
    return np.random.rand(num_cities, 2)

# Calculate the total distance of a route
def total_distance(route, dist_matrix):
    return sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)) + dist_matrix[route[-1], route[0]]

# Create initial population
def create_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

# Evaluate fitness of the population
def evaluate_population(population, dist_matrix):
    return [total_distance(individual, dist_matrix) for individual in population]

# Selection: Select parents using tournament selection
def select_parents(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        selected.append(min(tournament, key=lambda x: x[1])[0])
    return selected

# Crossover: Ordered crossover
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    ptr = end
    for gene in parent2:
        if gene not in child:
            if ptr >= size:
                ptr = 0
            child[ptr] = gene
            ptr += 1
    return child

# Mutation: Swap mutation
def mutate(individual, mutation_rate=0.01):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Genetic Algorithm for TSP
def genetic_algorithm_tsp(cities, pop_size=100, num_generations=500, mutation_rate=0.01):
    dist_matrix = distance_matrix(cities, cities)
    population = create_population(pop_size, len(cities))

    for generation in range(num_generations):
        fitness = evaluate_population(population, dist_matrix)
        parents = select_parents(population, fitness)
        next_population = []

        for i in range(0, pop_size, 2):
            parent1, parent2 = random.sample(parents, 2)
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent2, parent1), mutation_rate)
            next_population.extend([child1, child2])

        population = next_population

    best_route = min(population, key=lambda ind: total_distance(ind, dist_matrix))
    return best_route, total_distance(best_route, dist_matrix)

# Local Search Optimization for TSP
def local_search_tsp(cities, max_iterations=1000):
    dist_matrix = distance_matrix(cities, cities)

    def get_neighbors(route):
        neighbors = []
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                neighbor = route[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors

    def evaluate(route):
        return total_distance(route, dist_matrix)

    current_route = random.sample(range(len(cities)), len(cities))
    current_distance = evaluate(current_route)

    for iteration in range(max_iterations):
        neighbors = get_neighbors(current_route)
        next_route = min(neighbors, key=evaluate)
        next_distance = evaluate(next_route)

        if next_distance < current_distance:
            current_route = next_route
            current_distance = next_distance
        else:
            break

    return current_route, current_distance

# Function to plot the cities and the route
def plot_route(cities, route, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='red')

    for i in range(len(route)):
        plt.plot([cities[route[i], 0], cities[route[(i + 1) % len(route)], 0]],
                 [cities[route[i], 1], cities[route[(i + 1) % len(route)], 1]], 'b-')

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# Generate random cities
num_cities = 20
cities = create_cities(num_cities)

# Solve TSP using Genetic Algorithm
best_route_ga, best_distance_ga = genetic_algorithm_tsp(cities)
print("Best route (GA):", best_route_ga)
print("Best distance (GA):", best_distance_ga)

# Solve TSP using Local Search Optimization
best_route_lso, best_distance_lso = local_search_tsp(cities)
print("Best route (LSO):", best_route_lso)
print("Best distance (LSO):", best_distance_lso)

# Plot the routes
plot_route(cities, best_route_ga, f"Genetic Algorithm (Distance: {best_distance_ga:.2f})")
plot_route(cities, best_route_lso, f"Local Search Optimization (Distance: {best_distance_lso:.2f})")



































































































































































































































































































































































































































































