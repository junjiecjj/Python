

# https://mp.weixin.qq.com/s?__biz=Mzg3ODY4ODE5MQ==&mid=2247485887&idx=1&sn=d7c16be29800c8331df221d7670075a2&chksm=ce759c8e0747f9b7adc4cef138ec2c0b742341c743d3ff9e0371e7f3090dc0696ea09eab2e6e&mpshare=1&scene=1&srcid=0901AKZIPZmhAmpFmEYqM3E4&sharer_shareinfo=ce1629539c1d861e6edbe806f062df45&sharer_shareinfo_first=ce1629539c1d861e6edbe806f062df45&exportkey=n_ChQIAhIQxBie%2F3pbSxzaq6g%2BPMYTwxKfAgIE97dBBAEAAAAAAEiLLn%2Fuj0QAAAAOpnltbLcz9gKNyK89dVj0nDiK6TTahl%2BK1GWAczTX3ie5AnztzgEl6Gfh6KBj0izqHu8clVtexLJ2b37viGdCUsnfHdMJXBe7OWk9UssbMaMCci9jQgZOJu%2FhbHAbF17UUq%2FbzHG2jubko%2B5VTuKSveG3wANkcPuWNEUzFRewOgvPltLPrWEq0MOSm638nj7l9vdQKlJAZTrh6kgstxYFQpNs0Fa%2FZHZPhfqPgm8htDsmCvq1b1DPxWSINoIpTTkXqjuNGD1QyNXX5Kghh89Qte2Ea7dvg0VxCvdj7wgX0INsuJbt2d9ldh0AgbEnUKgvZ8EUUlUa%2BdsCDKQvDSxmleRQljeJUDYT&acctmode=0&pass_ticket=%2FIvTbhYj8v5K56tSt3%2FdHuxzs3RT56tmteh1Igc3rWxMjkm4sVAxOw81hNzZWyxf&wx_header=0#rd

import random
import math

distances = [
    [0, 2, 5, 7],
    [2, 0, 6, 3],
    [5, 6, 0, 8],
    [7, 3, 8, 0]
]

numAnts = 5
numCities = 4
maxIterations = 100
evaporationRate = 0.5
alpha = 1.0
beta = 2.0

pheromones = []

def main():
    random.seed()
    initialize_pheromones()

    for iter in range(maxIterations):
        antPaths = []

        for ant in range(numAnts):
            path = generate_ant_path()
            antPaths.append(path)
            update_pheromones(path)

        evaporate_pheromones()

        bestPath = antPaths[0]
        print(f"Iteration {iter + 1}: Best Path -> {' -> '.join(map(str, bestPath))}")

def initialize_pheromones():
    global pheromones
    pheromones = [[1.0] * numCities for _ in range(numCities)]

def generate_ant_path():
    startCity = 0
    path = [startCity]

    while len(path) < numCities:
        currentCity = path[-1]
        nextCity = choose_next_city(currentCity, path)
        path.append(nextCity)

    return path

def choose_next_city(currentCity, path):
    availableCities = [city for city in range(numCities) if city not in path]

    probabilities = []
    totalProbability = 0.0

    for nextCity in availableCities:
        pheromone = math.pow(pheromones[currentCity][nextCity], alpha)
        distance = 1.0 / distances[currentCity][nextCity]
        probability = pheromone * distance
        probabilities.append(probability)
        totalProbability += probability

    probabilities = [probability / totalProbability for probability in probabilities]

    randomValue = random.random()
    cumulativeProbability = 0.0

    for i, probability in enumerate(probabilities):
        cumulativeProbability += probability
        if randomValue <= cumulativeProbability:
            return availableCities[i]

    return availableCities[-1]

def update_pheromones(path):
    pheromoneDeposit = 1.0

    for i in range(len(path) - 1):
        currentCity = path[i]
        nextCity = path[i + 1]
        pheromones[currentCity][nextCity] += pheromoneDeposit
        pheromones[nextCity][currentCity] += pheromoneDeposit

def evaporate_pheromones():
    for i in range(numCities):
        for j in range(numCities):
            pheromones[i][j] *= (1.0 - evaporationRate)

if __name__ == '__main__':
    main()













































































































































































































































































