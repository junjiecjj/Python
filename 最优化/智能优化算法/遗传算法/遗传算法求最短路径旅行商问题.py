#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:28:02 2024

@author: jack

https://blog.csdn.net/KOBEYU652453/article/details/114787101


"""


from math import floor
import numpy as np
import matplotlib.pyplot as plt  # 导入所需要的库

class Gena_TSP(object):
    def __init__(self ,data ,maxgen=200 ,size_pop=200 ,cross_prob=0.9 ,pmuta_prob=0.01 ,select_prob=0.8):
        self.maxgen = maxgen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.pmuta_prob = pmuta_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.data = data  # 城市的左边数据
        self.num =len(data)  # 城市个数 对应染色体长度
        # 距离矩阵n*n, 第[i,j]个元素表示城市i到j距离matrix_dis函数见下文
        self.matrix_distance = self.matrix_dis()

        # 通过选择概率确定子代的选择个数
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)

        # 父代和子代群体的初始化（不直接用np.zeros是为了保证单个染色体的编码为整数，np.zeros对应的数据类型为浮点型）
        self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop, self.num)#父 print(chrom.shape)(200, 14)
        self.sub_sel = np.array([0] * self.select_num * self.num).reshape(self.select_num, self.num)#子 (160, 14)

        # 存储群体中每个染色体的路径总长度，对应单个染色体的适应度就是其倒数  #print(fitness.shape)#(200,)
        self.fitness = np.zeros(self.size_pop)


        self.best_fit = []##最优距离
        self.best_path = []#最优路径

    # 计算城市间的距离函数  n*n, 第[i,j]个元素表示城市i到j距离
    def matrix_dis(self):
        res = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(i + 1, self.num):
                res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])#求二阶范数 就是距离公式
                res[j, i] = res[i, j]
        return res

    # 随机产生初始化群体函数
    def rand_chrom(self):
        rand_ch = np.array(range(self.num)) ## num 城市个数 对应染色体长度  =14
        for i in range(self.size_pop):#size_pop  # 群体个数 200
            np.random.shuffle(rand_ch)#打乱城市染色体编码
            self.chrom[i, :] = rand_ch
            self.fitness[i] = self.comp_fit(rand_ch)

    # 计算单个染色体的路径距离值，可利用该函数更新fittness
    def comp_fit(self, one_path):
        res = 0
        for i in range(self.num - 1):
            res += self.matrix_distance[one_path[i], one_path[i + 1]] #matrix_distance n*n, 第[i,j]个元素表示城市i到j距离
        res += self.matrix_distance[one_path[-1], one_path[0]]#最后一个城市 到起点距离
        return res

    # 路径可视化函数
    def out_path(self, one_path):
        res = str(one_path[0] + 1) + '-->'
        for i in range(1, self.num):
            res += str(one_path[i] + 1) + '-->'
        res += str(one_path[0] + 1) + '\n'
        print(res)

    # 子代选取，根据选中概率与对应的适应度函数，采用随机遍历选择方法
    def select_sub(self):
        fit = 1. / (self.fitness)  # 适应度函数
        cumsum_fit = np.cumsum(fit)#累积求和   a = np.array([1,2,3]) b = np.cumsum(a) b=1 3 6
        pick = cumsum_fit[-1] / self.select_num * (np.random.rand() + np.array(range(self.select_num)))#select_num  为子代选择个数 160
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]#chrom 父

    # 交叉，依概率对子代个体进行交叉操作
    def cross_sub(self):
        if self.select_num % 2 == 0:#select_num160
            num = range(0, self.select_num, 2)
        else:
            num = range(0, self.select_num - 1, 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1, :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i + 1, :])

    def intercross(self, ind_a, ind_b):#ind_a，ind_b 父代染色体 shape=(1,14) 14=14个城市
        r1 = np.random.randint(self.num)#在num内随机生成一个整数 ，num=14.即随机生成一个小于14的数
        r2 = np.random.randint(self.num)
        while r2 == r1:#如果r1==r2
            r2 = np.random.randint(self.num)#r2重新生成
        left, right = min(r1, r2), max(r1, r2)#left 为r1,r2小值 ，r2为大值
        ind_a1 = ind_a.copy()#父亲
        ind_b1 = ind_b.copy()# 母亲
        for i in range(left, right + 1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[i] = ind_b1[i]#交叉 （即ind_a  （1,14） 中有个元素 和ind_b互换
            ind_b[i] = ind_a1[i]
            x = np.argwhere(ind_a == ind_a[i])
            y = np.argwhere(ind_b == ind_b[i])

            """
                   下面的代码意思是 假如 两个父辈的染色体编码为【1234】，【4321】
                   交叉后为【1334】，【4221】
                   交叉后的结果是不满足条件的，重复个数为2个
                   需要修改为【1234】【4321】（即修改会来
                   """
            if len(x) == 2:
                ind_a[x[x != i]] = ind_a2[i]#查找ind_a 中元素=- ind_a[i] 的索引
            if len(y) == 2:
                ind_b[y[y != i]] = ind_b2[i]
        return ind_a, ind_b

    # 变异模块  在变异概率的控制下，对单个染色体随机交换两个点的位置。
    def mutation_sub(self):
        for i in range(self.select_num):#遍历每一个 选择的子代
            if np.random.rand() <= self.cross_prob:#如果随机数小于变异概率
                r1 = np.random.randint(self.num)#随机生成小于num==14 的数
                r2 = np.random.randint(self.num)
                while r2 == r1:#如果相同
                    r2 = np.random.randint(self.num)#r2再生成一次
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i, [r2, r1]]#随机交换两个点的位置。

    # 进化逆转  将选择的染色体随机选择两个位置r1:r2 ，将 r1:r2 的元素翻转为 r2:r1 ，如果翻转后的适应度更高，则替换原染色体，否则不变
    def reverse_sub(self):
        for i in range(self.select_num):#遍历每一个 选择的子代
            r1 = np.random.randint(self.num)#随机生成小于num==14 的数
            r2 = np.random.randint(self.num)
            while r2 == r1:#如果相同
                r2 = np.random.randint(self.num)#r2再生成一次
            left, right = min(r1, r2), max(r1, r2)#left取r1 r2中小值，r2取大值
            sel = self.sub_sel[i, :].copy()#sel 为父辈染色体 shape=（1,14）

            sel[left:right + 1] = self.sub_sel[i, left:right + 1][::-1]#将染色体中(r1:r2)片段 翻转为（r2:r1)
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i, :]):#如果翻转后的适应度小于原染色体，则不变
                self.sub_sel[i, :] = sel

    # 子代插入父代，得到相同规模的新群体
    def reins(self):
        index = np.argsort(self.fitness)[::-1]#替换最差的（倒序）
        self.chrom[index[:self.select_num], :] = self.sub_sel


#路径坐标
data = np.array([16.47,96.10,16.47,94.44,20.09,92.54,
    22.39,93.37,25.23,97.24,22.00,96.05,20.47,97.02,
    17.20,96.29,16.30,97.38,14.05,98.12,16.53,97.38,
    21.52,95.59,19.41,97.13,20.09,92.55]).reshape(14,2)
def main(data):
    Path_short = Gena_TSP(data)  # 根据位置坐标，生成一个遗传算法类
    Path_short.rand_chrom()  # 初始化父类

    ## 绘制初始化的路径图
    fig, ax = plt.subplots()
    x = data[:, 0]
    y = data[:, 1]
    ax.scatter(x, y, linewidths=0.1)
    for i, txt in enumerate(range(1, len(data) + 1)):
        ax.annotate(txt, (x[i], y[i]))
    res0 = Path_short.chrom[0]
    x0 = x[res0]
    y0 = y[res0]
    for i in range(len(data) - 1):
        plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
    plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color='r', width=0.005, angles='xy', scale=1,
               scale_units='xy')
    plt.show()
    print('初始染色体的路程: ' + str(Path_short.fitness[0]))

    # 循环迭代遗传过程
    for i in range(Path_short.maxgen):
        Path_short.select_sub()  # 选择子代
        Path_short.cross_sub()  # 交叉
        Path_short.mutation_sub()  # 变异
        Path_short.reverse_sub()  # 进化逆转
        Path_short.reins()  # 子代插入

        # 重新计算新群体的距离值
        for j in range(Path_short.size_pop):
            Path_short.fitness[j] = Path_short.comp_fit(Path_short.chrom[j, :])

        # 每隔三十步显示当前群体的最优路径
        index = Path_short.fitness.argmin()
        if (i + 1) % 30 == 0:
            print('第' + str(i + 1) + '步后的最短的路程: ' + str(Path_short.fitness[index]))
            print('第' + str(i + 1) + '步后的最优路径:')
            Path_short.out_path(Path_short.chrom[index, :])  # 显示每一步的最优路径

        # 存储每一步的最优路径及距离
        Path_short.best_fit.append(Path_short.fitness[index])
        Path_short.best_path.append(Path_short.chrom[index, :])

    res1 = Path_short.chrom[0]
    x0 = x[res1]
    y0 = y[res1]
    for i in range(len(data) - 1):
        plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
    plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color='r', width=0.005, angles='xy', scale=1,
               scale_units='xy')
    plt.show()

    return Path_short  # 返回遗传算法结果类

Path_short=main(data)
print(Path_short)

