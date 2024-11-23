#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:57:29 2024

@author: jack
"""

import galois
import numpy as np
import itertools


# GF = galois.GF(2**2, repr = "poly")
GF = galois.GF(2**3, repr = "int")
# GF = galois.GF(2**3, repr = "power")

# 访问有限域的不可约多项式
print(GF.irreducible_poly)

# 访问有限域的属性
print(GF.properties)

# 检索有限域的元素
print(GF.elements)

# 具体有限域的最小生成元
print(GF.primitive_element)

GF = galois.GF(2**3, repr = "int")
a_poly = galois.Poly([1, 2, 3, 4], field = GF)
print(a_poly)

b_poly = galois.Poly([1, 1],  field = GF)
print(b_poly)

p = GF.characteristic
print(p)

a = GF(3)
b = GF(5)
a_int = 3
b_int = 5
print((a_int + b_int) % p)
print(a + b)

print((a_int - b_int) % p)
print(a - b)

print((a_int * b_int) % p)
print(a * b)

print()
print()

print()
print()


# 输出有限域的算数表(加减乘）
# 以加法为例:
print(GF.arithmetic_table("+"), '\n')
print(GF.arithmetic_table("-"), '\n')
print(GF.arithmetic_table("*"), '\n')
print(GF.arithmetic_table("/"), '\n')

# 输出有限域的生成元表
print(GF.repr_table())
#  Power    Polynomial     Vector    Integer
# ------- ------------- ----------- ---------
#    0          0        [0, 0, 0]      0
#   x^0         1        [0, 0, 1]      1
#   x^1         x        [0, 1, 0]      2
#   x^2        x^2       [1, 0, 0]      4
#   x^3       x + 1      [0, 1, 1]      3
#   x^4      x^2 + x     [1, 1, 0]      6
#   x^5    x^2 + x + 1   [1, 1, 1]      7
#   x^6      x^2 + 1     [1, 0, 1]      5

# 构建有限域上的多项式
f = galois.Poly([1, 1, 1, 1], field = GF)
print(f)

# 以多项式形式查看有限域上的元素
x1 = [23, 123, 56, 64]
GF = galois.GF(2**8, repr = "poly")
print(GF(x1))

# 以int形式查看有限域上的元素, default is int
x1 = [23, 123, 56, 64]
GF = galois.GF(2**8, repr = "int")
print(GF(x1))

# 有限域上加法和乘法
# 对于单个元素的加法和乘法
x = GF(34)
y = GF(23)
c = x + y
print(c)
d = x * y
print(d)

GF.repr("int")
x = GF(34)
isinstance(x, GF)
# Out[23]: True

isinstance(x, galois.FieldArray)
# Out[24]: True

isinstance(x, galois.Array)
# Out[25]: True

isinstance(x, np.ndarray)
# Out[26]: True


# 对于整个数组进行的加法或乘法
x = GF([184, 25, 157, 31])
y = GF([179, 9, 139, 27])
c = x + y
print(c)
c = x - y
print(c)
c = x * y
print(c)
print(c.sum())
c = x / y
print(c)

# 复杂的算术，如平方根和对数底
c = np.sqrt(x)
print(c)
c = np.log(x)
print(c)


x = GF([184, 25, 157, 31])
y = GF([179, 9, 139, 27])
a = np.dot(x, y)  # a = c.sum()
print(a)

GF.repr("int")
print(x)

# create an identity matrix using Identity().
I = GF.Identity(4)
print(I)
# generate a random array of given shape call Random().
r = GF.Random((3, 2), seed = 1)
print(r)


GF = galois.GF(2**1, repr = "int")
for i in GF.elements:
    for j in GF.elements:
        print(f"{i} + {j} = {i+j}")

# np.array(list(itertools.product(GF.elements, repeat = 3)))
res = {}
for i in range(2**3):
    res[i] = []
for comb in itertools.product(GF.elements, repeat = 2):
    # res[int(i+j+k)] = []
    res[int(np.sum(GF([int(g) for g in comb])))].append([int(g) for g in comb])

##################
import commpy as comm

l = 2
uu = np.random.randint(0, 2, size = (l, 10))
q = 2**l
GF = galois.GF(2**l, repr = "int")
I = GF.Identity(l)
I[0] = 1
# I = [1]*p
# I = GF(I)

print(f"uu = \n  {uu}")
print(f"I@GF(uu) = \n  {I@GF(uu)}")

qbits = comm.utilities.dec2bitarray(np.arange(q), l).reshape(-1, l).T
qfun = I @ GF(qbits)
print(f"qbits = \n  {qbits}")
print(f"I@GF(qbits) = \n  {I@GF(qbits)}")































































































