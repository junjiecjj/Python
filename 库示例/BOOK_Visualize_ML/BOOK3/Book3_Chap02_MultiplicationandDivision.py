#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:36:25 2025

@author: jack
"""



#%% Bk3_Ch2_01

num1 = 2
num2 = 3

# add two numbers
prod = num1*num2

# display the computation
print('The product of {0} and {1} is {2}'.format(num1, num2, prod))



#%% Bk3_Ch2_02

# define a function to calculate factorial

num = int(input("Enter an integer: "))

factorial = 1

# check if the number is negative, positive or zero
if num < 0:
    print("Factorial does not exist for negative numbers")
elif num == 0:
    print("The factorial of 0 is ", factorial)
else:
    for i in range(1,num + 1):
       factorial = factorial*i
    print("The factorial of",num," is ",factorial)


#%% Bk3_Ch2_03

import numpy as np

a_i = np.linspace(1,10,10)
print(a_i)

a_i_cumprod = np.cumprod(a_i)
np.set_printoptions(suppress=True)
print(a_i_cumprod)

#%% Bk3_Ch2_04

num1 = 6
num2 = 3

# add two numbers
division = num1/num2

# display the computation
print('The division of {0} over {1} is {2}'.format(num1, num2, division))

#%% Bk3_Ch2_05

num1 = 7
num2 = 2

# add two numbers
remainder = num1%num2

# display the computation
print('The remainder of {0} over {1} is {2}'.format(num1, num2, remainder))


#%% Bk3_Ch2_06

import numpy as np

# define a column vector
a_col = np.array([[1], [2], [3]])

b_col = 2*a_col

# define a row vector
a_row = np.array([[1, 2, 3]])

b_row = 2*a_row

# define a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = 2*A


#%% Bk3_Ch2_07

import numpy as np

# row vector dot product

a_row = np.array([[1, 2, 3]])
b_row = np.array([[4, 3, 2]])

a_dot_b = np.inner(a_row, b_row)

print(a_dot_b)
print(np.inner(a_row[:], b_row[:]))
print(np.sum(a_row * b_row))

#%% column vector dot product

a_col = np.array([[1],  [2], [3]])
b_col = np.array([[-1], [0], [1]])

a_dot_b = np.inner(a_col, b_col)
print(a_dot_b) # tensor product

print(np.sum(a_col * b_col))



#%% Bk3_Ch2_08

import numpy as np
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])

# calculate element-wise product of row vectors
a_times_b = a*b

A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[1, 2, 3],
              [-1, 0, 1]])

# calculate element-wise product of matrices
A_times_B = A*B



#%% Bk3_Ch2_09

import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[4, 2],
              [3, 1]])

# calculate matrix multiplication
C = A@B



#%% Bk3_Ch2_10_A

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Repeatability
np.random.seed(7)

# Generate matrix A and B
m = 6
p = 3
n = 4

A = np.random.uniform(-1,1,m*p).reshape(m, p)
B = np.random.uniform(-1,1,p*n).reshape(p, n)

C = A@B

all_max = 1
all_min = -1

#  matrix multiplication, first perspective

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(A,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"},
                 yticklabels=np.arange(1,m+1), xticklabels=np.arange(1,p+1))
ax.set_aspect("equal")
plt.title('$A$')
plt.yticks(rotation=0)

plt.sca(axs[1])
plt.title('$@$')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(B,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"},
                 yticklabels=np.arange(1,p+1), xticklabels=np.arange(1,n+1))
ax.set_aspect("equal")
plt.title('$B$')
plt.yticks(rotation=0)

plt.sca(axs[3])
plt.title('$=$')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(C,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"},
                 yticklabels=np.arange(1,m+1), xticklabels=np.arange(1,n+1))
ax.set_aspect("equal")
plt.title('$C$')
plt.yticks(rotation=0)

#%% Bk3_Ch2_11

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import seaborn as sns

# Repeatability
np.random.seed(0)
# Generate matrix A
n = 4
A = np.random.uniform(-1.5,1.5,n*n).reshape(n, n)

all_max = 1.5
all_min = -1.5

# matrix inverse
A_inverse = inv(A)

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(A,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"},
                 yticklabels=np.arange(1,n+1), xticklabels=np.arange(1,n+1),
                 annot = True,fmt=".2f")
ax.set_aspect("equal")
plt.title('$A$')
plt.yticks(rotation=0)

plt.sca(axs[1])
plt.title('$@$')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(A_inverse,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"},
                 yticklabels=np.arange(1,n+1), xticklabels=np.arange(1,n+1),
                 annot = True,fmt=".2f")
ax.set_aspect("equal")
plt.title('$A^{-1}$')
plt.yticks(rotation=0)

plt.sca(axs[3])
plt.title('$=$')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(A@A_inverse,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"},
                 yticklabels=np.arange(1,n+1), xticklabels=np.arange(1,n+1),
                 annot = True,fmt=".2f")
ax.set_aspect("equal")
plt.title('$I$')
plt.yticks(rotation=0)










#%%








#%%








#%%








#%%








#%%








#%%








#%%








#%%








#%%








