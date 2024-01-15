#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:16:57 2023

@author: jack
https://blog.csdn.net/a254342594/article/details/82526387

https://zhuanlan.zhihu.com/p/129373740

最短路径就是从一个指定的顶点出发，计算从该顶点出发到其他所有顶点的最短路径。通常用Dijkstra算法，Floyd算法求解(每对顶点之间的最短路径用Floyd算法)。
主要介绍以下几种算法：
    Dijkstra最短路算法（单源最短路）
    Bellman–Ford算法（解决负权边问题）
    SPFA算法（Bellman-Ford算法改进版本）
    Floyd最短路算法（全局/多源最短路）


Dijkstra算法是一种贪心算法，用于解决带权重的有向图或无向图中的单源最短路径问题。注意：可以是无向图或有向图。

Dijkstra算法:
(1) 从源点开始，每次选择当前距离源点最近的一个未标记节点，
(2) 然后更新与该节点相邻的节点的距离，
(3) 直到所有节点标记完毕，
(4) 最短路径即可得到。


(1) 找出最便宜的节点，即可在最短时间内前往的节点。
(2) 对于该节点的邻居，检查是否有前往它们的更短路径，如果有，就更新其开销；
(3) 重复(1)(2)，直到对图中的每个节点都这样做了；
(4) 计算最终路径。


"""

# Python实现Dijkstra算法

import heapq

def Dijkstra(graph, start):
    # 初始化距离字典，用于记录每个节点到起点的距离
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    print(f"0, {dist}")
    # 初始化堆
    heap = []
    heapq.heappush(heap, (0, start))
    i = 0
    # 循环堆
    while heap:
        i += 1
        (distance, current_node) = heapq.heappop(heap)
        print(f"{i}: pop {current_node}-{distance}, {heap}")
        # 当前节点已经求出最短路径
        if distance > dist[current_node]:
            continue
        # 遍历当前节点的相邻节点
        for neighbor, weight in graph[current_node].items():
            dist_neighbor = dist[current_node] + weight
            print(f"{i}: {current_node}->{neighbor}={weight}, {dist_neighbor} ")
            # 更新最短路径距离
            if dist_neighbor < dist[neighbor]:
                dist[neighbor] = dist_neighbor
                heapq.heappush(heap, (dist_neighbor, neighbor))  # 入堆的是当前节点的邻居节点，及距离(当前节点的距离+当前节点到邻居节点的距离，此和不一定是邻居的最短路径)。
                print(f"{i}: {dist}, {heap}")
        print("\n")
    return dist

# 测试代码
graph = {'A': {'B': 5, 'C': 1},
          'B': {'A': 5, 'C': 2, 'D': 1},
          'C': {'A': 1, 'B': 2, 'D': 4, 'E': 8},
          'D': {'B': 1, 'C': 4, 'E': 3, 'F': 6},
          'E': {'C': 8, 'D': 3},
          'F': {'D': 6}}
graph = {'A': {'B': 5, 'C': 1},
         'B': {'D': 1},
         'C': {'B': 2, 'D': 4, 'E': 8},
         'D': {'F': 6},
         'E': {'D': 3},
         'F': {}}

print(Dijkstra(graph, 'A'))







# https://blog.csdn.net/weixin_59450364/article/details/124115888

import numpy as np
import copy

def main():
    #无穷大
    infinity = float('inf')
    a = infinity
    #构建邻接矩阵
    adjacency_matrix = np.array([[a,6,5,3,a,a],
                                 [a,a,a,1,a,3],
                                 [a,a,a,1,2,a],
                                 [a,a,a,a,9,7],
                                 [a,a,a,a,a,5],
                                 [a,a,a,a,a,a]])
    #构建距离数组
    dist = np.array([0,6,5,3,a,a])
    #构建前驱数组
    precursor = np.array([-1,1,1,1,-1,-1])
    #初始集合
    S = {1}
    V = {1,2,3,4,5,6}
    V_subtract_S = V - S

    for i in range(len(V_subtract_S)-1):
        dist_copy = []
        V_subtract_S_list = list(V_subtract_S)
        for j in V_subtract_S:
            dist_copy.append(dist[j - 1])
        min_index = dist_copy.index(min(dist_copy))  # 查找dist_copy中最小的元素的位置
        S.add(V_subtract_S_list[min_index])
        current_node = V_subtract_S_list[min_index]
        V_subtract_S = V - S

        for j in V_subtract_S:
            dist_copy.append(dist[j - 1])
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[current_node-1][j] < a:
                if dist[current_node-1] + adjacency_matrix[current_node-1][j] < dist[j]:
                    dist[j] = dist[current_node-1] + adjacency_matrix[current_node-1][j]
                    precursor[j] = current_node

    #打印最佳路径
    temp = 1
    path = []
    path.insert(0, 6)
    precursor = list(precursor)
    front_code = precursor[5]
    while temp:
        path.insert(0,front_code)
        #front_code的数字对应的另一个节点
        front_code_index = path[0] - 1
        front_code = precursor[front_code_index]
        if front_code == 1:
            temp = 0
    path.insert(0,1)
    for i in path:
        if i == 1:
            path[path.index(i)] = 's'
        if i == 2:
            path[path.index(i)] = 'v'
        if i == 3:
            path[path.index(i)] = 'u'
        if i == 4:
            path[path.index(i)] = 'w'
        if i == 5:
            path[path.index(i)] = 'z'
        if i == 6:
            path[path.index(i)] = 't'

    print(path)























