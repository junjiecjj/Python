#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:55:18 2024

@author: jack
"""


import numpy as np
import math
import cvxpy as cpy




def SDRsolver(G, hr, hd, N):
    Phai = np.diag(hr.conjugate()) @ G
    A = Phai @ (Phai.T.conjugate())
    B = (Phai @ hd.reshape(-1, 1))
    C = (hd.T.conjugate()) @ (Phai.T.conjugate())
    C = np.append(C, 0).reshape(1, -1)
    R = np.concatenate((A, B), axis = 1)
    R = np.concatenate((R, C), axis = 0)

    ## use cvx to solve
    V = cpy.Variable((N+1, N+1), symmetric = True)
    obj = cpy.Maximize(cpy.real(cpy.trace(R@V)) + cpy.norm(hd, 2))
    # The operator >> denotes matrix inequality.
    constraints = [
                    0 << V,
                   cpy.diag(V) == 1,
                   ]
    prob = cpy.Problem(obj, constraints)
    prob.solve()

    if prob.status=='optimal':
         print("optimal")
    else:
          print("Not optimal")


    v = 1
    return prob.value, v




















