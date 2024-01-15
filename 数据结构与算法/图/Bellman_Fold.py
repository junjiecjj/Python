#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:50:01 2023

@author: jack
Bellman-Ford算法是一种动态规划算法，用于解决带权重的有向图或无向图中的单源最短路径问题，同时能够处理负权边。Bellman-Ford算法中，对于一条边(u, v)，先从源点s到u的最短路径dist[u]已经求得，通过松弛操作，可以尝试更新从源点s到v的最短路径dist[v]，如果更新成功，则表示有一条更短的路径从源点s到v。

需要注意的是，Bellman-Ford算法不能处理存在负权环的情况，因为负权环中的最短路径不存在。因此，Bellman-Ford算法通过检查是否存在负权环来判断是否有解。如果存在负权环，则说明最短路径无法计算。


"""

# Python实现Bellman-Ford算法

def bellman_ford(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0

    # 进行V-1次松弛操作
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u].items():
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight

    # 检查负权环
    for u in graph:
        for v, weight in graph[u].items():
            if dist[u] + weight < dist[v]:
                raise ValueError('Graph contains a negative weight cycle')

    return dist

# 测试代码
graph = {'A': {'B': 5, 'C': 1},
         'B': {'A': 5, 'C': 2, 'D': 1},
         'C': {'A': 1, 'B': 2, 'D': 4, 'E': 8},
         'D': {'B': 1, 'C': 4, 'E': -3, 'F': 6},
         'E': {'C': 8, 'D': -3},
         'F': {'D': 6}}
print(bellman_ford(graph, 'A'))
