#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:42:18 2025

@author: jack
"""

#%% Bk3_Ch4_01

# set A: odd
A = set([1,3,5])

# set B: even
B = set([2,4,6])

# set C: less than 4
C = set([1,2,3])

# A union B
A_union_B = A.union(B) #A|B, A or B

# A intersects (meets) B
A_meet_B = A.intersection(B) #A&B, A and B

# A minus B
A_minus_B = A.difference(B) # A - B, set difference

# A union C
A_union_C = A.union(C) #A|C, A or C

# A intersects (meets) C
A_meet_C = A.intersection(C) #A&C, A and C

# A minus C
A_minus_C = A.difference(C) # A - C, set difference

# C minus A
C_minus_A = C.difference(A) # C - A, set difference





#%% Bk3_Ch4_02_A

from sympy.abc import x,y

expr_x = x**3 + 2*x**2 - x - 2

print(expr_x.subs(x,1))

# Bk_Ch4_02_B

from sympy import cos

expr_cos_y = expr_x.subs(x,cos(y))

print(expr_cos_y)

# Bk_Ch4_02_C

from sympy import symbols

x,y = symbols('x,y')
expr_1 = x + y
print(expr_1)

x1,x2 = symbols('x1,x2')
expr_2 = x1 + x2
print(expr_2)


#%% Bk3_Ch4_03

from sympy import *
x,y,z = symbols('x y z')

# simplify mathematical expressions
expr_1 = sin(x)**2 - cos(x)**2
print(simplify(expr_1))

# expand polynomial expressions
expr_2 = (x + 1)**3
print(expand(expr_2))

# take a polynomial and factors it into irreducible factors
expr_3 = x**3 + 2*x**2 - x - 2
print(factor(expr_3))

# collect common powers of a term in an expression
expr_collected = collect(expr_3 - x**2 - 2*x, x)
print(expr_collected)

#%% Bk3_Ch4_04

from sympy.abc import x, y, z
expr = x**3 + 2*x**2 - x - 2

from sympy.utilities.lambdify import lambdify
f_x = lambdify(x, expr)

print(f_x(1))


#%% Bk3_Ch4_05

from sympy.abc import x, a
from sympy import Poly
import matplotlib.pyplot as plt
import numpy as np

for n in [4, 8, 12, 5, 9, 13, 36]:

    expr = (x + 1)**n

    expr_expand = expr.expand()
    expr_expand = Poly(expr_expand)

    poly_coeffs = expr_expand.coeffs()

    print(poly_coeffs)

    degrees = np.linspace(n,0,n + 1)

    fig, ax = plt.subplots()

    plt.stem(degrees, np.array(poly_coeffs, dtype=float))
    plt.xlim(0,n)
    plt.xticks(np.arange(0,n+1))
    y_max = max(poly_coeffs)
    y_max = float(y_max)
    plt.ylim(0,y_max)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.invert_xaxis()
    plt.xlabel('Degree')
    plt.ylabel('Coefficient')
    plt.show()


#%% Bk3_Ch4_06

import itertools

letters = "ABC"

# find all combinations containing 2 letters

cmb = itertools.combinations(letters, 2)

for val in cmb:
    print(val)


#%% Bk3_Ch4_07

import itertools

letters = "ABC"

# Arranging 2 elements out of 3

per = itertools.permutations(letters, 2)

for val in per:
    print(val)


#%% Bk3_Ch4_08

import itertools

letters = "ABC"

# Arranging all 3 elements

per = itertools.permutations(letters)

for val in per:
    print(val)

#%% Bk3_Ch4_09

# use sympy to solve
from sympy.solvers import solve
from sympy import Symbol
x = Symbol('x')
roots = solve(-x**3 + x, x)

# use numpy to solve
import numpy as np

coeff = [-1, 0, 1, 0]
roots_V2 = np.roots(coeff)



#%%




#%%









