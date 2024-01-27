#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 00:50:20 2022
@author: jack

pip install gurobipy
conda install -c gurobi gurobi

1. 去 gurobi 官网注册下载Linux版本的压缩包，解压缩放在家目录下：/home/jack/gurobi1000/
2. 添加环境变量
    首先打开~/.bashrc，我这里用vim打开vim ~/.bashrc，之后在文件中添加(第一行的路径为gurobi存放路径，也就是你解压时候的位置)：
    export GUROBI_HOME="/home/jack/gurobi811/linux64"
    export PATH="${PATH}:${GUROBI_HOME}/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
    export GRB_LICENSE_FILE="/home/jack/gurobi811/gurobi.lic"
3. 在gurobi811/linux64目录下执行命令：python setup.py install
4.验证是否安装成功:cd  bin
                gurobi.sh

Gurobi的simplex比较强，解混合整数规划比较好。
MOSEK 的interior point比较好，解锥优化比较好，如SOCP, SDP.
"""


#===========================================================================
#  整数规划实例 1
# https://blog.csdn.net/sanpi329/article/details/106917171
#===========================================================================
import gurobipy as gob
# 8部电影
# 7个影厅
# 8个时段
I = list(range(8))  # 时段
J = list(range(7))  # 影厅
K = list(range(8))  # 电影

seat_j = [118, 86, 116, 85, 156, 142, 156]
# 一行为一个影厅,一列为一部电影
price_jk = [[60, 60, 65, 60, 65, 90, 60, 65],
            [65, 65, 85, 75, 60, 75, 85, 80],
            [60, 70, 75, 80, 75, 80, 80, 75],
            [65, 65, 80, 75, 80, 75, 75, 80],
            [60, 65, 65, 60, 75, 80, 80, 75],
            [60, 65, 65, 80, 75, 75, 80, 75],
            [60, 60, 75, 80, 75, 70, 60, 75]]
# 一行为一个时段,一列为一部电影
rate_ik =  [[0.50, 0.55, 0.45, 0.50, 0.60, 0.46, 0.55, 0.45],
            [0.42, 0.43, 0.41, 0.43, 0.45, 0.30, 0.53, 0.36],
            [0.58, 0.63, 0.67, 0.64, 0.70, 0.64, 0.54, 0.57],
            [0.62, 0.67, 0.70, 0.65, 0.75, 0.64, 0.53, 0.66],
            [0.65, 0.65, 0.73, 0.68, 0.75, 0.74, 0.67, 0.72],
            [0.66, 0.69, 0.78, 0.78, 0.78, 0.75, 0.74, 0.70],
            [0.67, 0.92, 0.87, 0.87, 0.75, 0.59, 0.68, 0.68],
            [0.67, 0.92, 0.87, 0.87, 0.75, 0.59, 0.68, 0.68]]
# 计算满座的票房二维列表,lt_all
all_jk = [[0 for col in K] for row in J]
for j in J:
    for k in K:
        all_jk[j][k] = price_jk[j][k] * seat_j[j]
# 创建模型
m = gob.Model("ass_mov")
# 创建变量.第i个时段在第j个影厅放映第k部电影
x = m.addVars(I, J, K, vtype=gob.GRB.BINARY)
# 更新变量环境
m.update()
# 创建目标函数
m.setObjective(sum(x[i, j, k] * rate_ik[i][k] * all_jk[j][k]
                for i in I for j in J for k in K),
                gob.GRB.MAXIMIZE)
# 创建约束条件约束条件
# 每部电影至少放映一次
m.addConstrs(sum(x[i,j,k] for i in I for j in J) >= 1 for k in K)
# 每个时段每个影厅只能放映一部电影
m.addConstrs(sum(x[i,j,k] for k in K) == 1 for i in I for j in J)
# 求解规划模型
m.optimize()

# 输出结果
result = [[0 for col in J] for row in I]
solution = m.getAttr('x',x)
# 得到排片矩阵
for k,v in solution.items():
    if v == 1:
        result[k[0]][k[1]] = k[2] + 1
# 得到最大收益值
max_get = sum(
    x[i, j, k].x * rate_ik[i][k] * all_jk[j][k]
    for i in I for j in J for k in K
)
# 打印最大收益值,和排片矩阵
print('最大收益为:',max_get)
print('最佳排片方法:')
print('\n影厅j|', J)
print('-'*28)
for idx,l in enumerate(result) :
    print(f'时段{idx}|', l)












































































