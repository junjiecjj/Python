#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:54:09 2022

https://leohope.com/%E8%A7%A3%E9%97%AE%E9%A2%98/2019/01/11/ldpc-with-python/

@author: jack
"""
import numpy as np


def genH(m, n, p):
    H = np.zeros((m, n))
    for i in range(m):
        # j=2*i
        for j in range(2*i,2*p+2*i):
            H[i, j%n] = 1
            j = j+1
    return H


# BPSK(c for codeword)
def modulate(c):
    for i in range(np.size(c)):
        if c[i] == 0:
            c[i] = 1
        else:
            c[i] = -1
    return c

def demodulate(y):
    for i in range(np.size(y)):
        if y[i] > 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

snr   = 6
n     = 1000
sigma = 10**(-snr/20)
noise = np.random.normal(0, sigma, size = n)



# LDPC decode(hard decision)
def decode(H,y,m,n,p):
    fr = np.zeros((m, 2 * p)) # check nodes received
    fs = np.zeros((m, 2 * p)) # check nodes send
    Sum = np.zeros(m) # check nodes received sum(for parity check)
    # message nodes table
    # 前p列为校验节点发来的消息，第p+1列为原始消息，第p+2列作为游标
    c=np.zeros((n,p+2))
    y1=np.zeros(n)

    # Fill the check nodes received table
    for i in range(m):
        count=0
        for j in range(n):
            if H[i][j] == 1:
                fr[i,count]=y[j]
                Sum[i]=Sum[i]+y[j]
                count = count+1

    # Calculate the check nodes send table
    for i in range(m):
        for j in range(2*p):
            fs[i,j]=(Sum[i]-fr[i,j])%2

    # Fill the message node table
    for i in range(m):
        count=0
        for j in range(n):
            if H[i][j]==1:
                index=int(c[j,p+1])
                c[j,index]=fs[i,count]
                count = count+1
                c[j,p+1]+=1

    # Fill the last column with y
    for i in range(n):
        c[i, p] = y[i]

    # Decision
    for i in range(n):
        count=0
        for j in range(p+1):
            if c[i,j] == 1:
                count+=1
        if count > (p+1)/2:
            y1[i]=1
    return y1


m=4 # Number of rows
n=8 # Number of columns
p=2 # Number of 1s in a colomn

H = genH(m, n, p)

y=np.zeros(n)
y=[1,1,0,1,0,1,0,1]

y1 = decode(H,y,m,n,p)
print(y1)

def errorRate(c,y1):
    err=0
    total=np.size(c)
    i=0
    for i in range(total):
        if c[i] != y1[i]:
            err+=1
    r = err/total
    return r
