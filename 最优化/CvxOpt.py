#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 22:32:26 2022

@author: jack
"""


from cvxopt import solvers,matrix




#=======================================================================
# CVXOPT 解 二次规划问题
# https://blog.csdn.net/Varalpha/article/details/106079698
#=======================================================================

p = matrix([[4., 1.], [1., 2.]])
q = matrix([1., 1.])
G = matrix([[-1.,0.],[0.,-1.]])
h = matrix([0.,0.]) # matrix里区分int和double，所以数字后面都需要加小数点
A = matrix([1., 1.], (1,2)) # A必须是一个1行2列
b = matrix(1.)

sol=solvers.qp(p, q, G, h, A, b)
print(sol['x'])



#=======================================================================
# CVXOPT 解 二次规划问题
# https://www.jianshu.com/p/df447c3e4efe
#=======================================================================
# 线性规划问题
import numpy as np
from cvxopt import matrix, solvers

A = matrix([[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0]])
b = matrix([1.0, -2.0, 0.0, 4.0])
c = matrix([2.0, 1.0])

sol = solvers.lp(c,A,b)

print(sol['x'])
print(np.dot(sol['x'].T, c))
print(sol['primal objective'])



#线性规划问题
import numpy as np
from cvxopt import matrix, solvers

A = matrix([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
b = matrix([2.0, 2.0, -2.0])
c = matrix([1.0, 2.0])
d = matrix([-1.0, -2.0])

sol1 = solvers.lp(c,A,b)
min = np.dot(sol1['x'].T, c)
sol2 = solvers.lp(d,A,b)
max = -np.dot(sol2['x'].T, d)

print('min=%s,max=%s'%(min[0][0], max[0][0]))

#二次型规划问题
from cvxopt import matrix, solvers

Q = 2*matrix([[2, .5], [.5, 1]])
p = matrix([1.0, 1.0])
G = matrix([[-1.0,0.0],[0.0,-1.0]])
h = matrix([0.0,0.0])
A = matrix([1.0, 1.0], (1,2))
b = matrix(1.0)

sol=solvers.qp(Q, p, G, h, A, b)
print(sol['x'])
print(sol['primal objective'])


#二次型规划问题
from cvxopt import matrix, solvers

P = matrix([[1.0, 0.0], [0.0, 0.0]])
q = matrix([3.0, 4.0])
G = matrix([[-1.0, 0.0, -1.0, 2.0, 3.0], [0.0, -1.0, -3.0, 5.0, 4.0]])
h = matrix([0.0, 0.0, -15.0, 100.0, 80.0])

sol=solvers.qp(P, q, G, h)
print(sol['x'])
print(sol['primal objective'])




#======================================
# https://zhuanlan.zhihu.com/p/410478494

import numpy as np
from cvxopt import matrix, solvers

A = matrix([[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0]])
b = matrix([1.0, -2.0, 0.0, 4.0])
c = matrix([2.0, 1.0])

sol = solvers.lp(c,A,b)

print(sol['x'])
print(np.dot(sol['x'].T, c))
print(sol['primal objective'])





import numpy as np
from cvxopt import matrix, solvers

A = matrix([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
b = matrix([2.0, 2.0, -2.0])
c = matrix([1.0, 2.0])
d = matrix([-1.0, -2.0])

sol1 = solvers.lp(c,A,b)
min = np.dot(sol1['x'].T, c)
sol2 = solvers.lp(d,A,b)
max = -np.dot(sol2['x'].T, d)

print('min=%s,max=%s'%(min[0][0], max[0][0]))

#======================================

# https://www.cnblogs.com/yijuncheng/p/11248633.html
from cvxopt import matrix, solvers
P = 2*matrix([ [2, .5], [.5, 1] ])
q = matrix([1.0, 1.0])
G = matrix([[-1.0,0.0],[0.0,-1.0]])
h = matrix([0.0,0.0])
A = matrix([1.0, 1.0], (1,2))
b = matrix(1.0)
sol=solvers.qp(Q, p, G, h, A, b)
print(sol['x'])




























































































































