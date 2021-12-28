# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:23:05 2017

@author: 科
"""

import numpy as np
import matplotlib.pyplot as plt
def loaddataset():
    datamat=[];labelmat=[]
    fr=open(r'E:\统计学习\MLiA_SourceCode\machinelearninginaction\Ch05\testSet.txt')
    for line in fr.readlines():
        linearr=line.strip().split('\t')
        datamat.append([1.0,float(linearr[0]),float(linearr[1])])
        labelmat.append(int(linearr[2]))
    return datamat,labelmat

def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

def gradascent(datamatin,classlabels):
    datamatrix=np.mat(datamatin)
    labelmat=np.mat(classlabels).transpose()
    m,n=np.shape(datamatrix)
    alpha=0.001
    maxcycles=500
    weights=np.ones((n,1))
    for i in np.arange(maxcycles):
        h=sigmoid(datamatrix*weights)
        error=(labelmat-h)
        weights=weights+alpha*datamatrix.transpose()*error
    return weights

def plotbestfit(weights):
    datamat,labelmat=loaddataset()
    dataarr=np.array(datamat)
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    n=(dataarr.shape)[0]
    for i in np.arange(n):
        if labelmat[i]==0:
            xcord1.append(dataarr[i,1])
            ycord1.append(dataarr[i,2])
        else:
            xcord2.append(dataarr[i,1])
            ycord2.append(dataarr[i,2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=10,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=5,c='blue')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('x1');plt.ylabel('x2')
    plt.show()

def randgradascent0(datamat,labelmat):
    m,n=np.shape(datamat)
    weights=np.ones(n)
    alpha=0.01
    for i in np.arange(m):
        h=sigmoid(sum(datamat[i]*weights))
        error=labelmat[i]-h
        weights=weights+alpha*error*(np.array(datamat[i]))
        print(weights)
    return weights
    
def randgradascent1(datamat,labelmat,itertimes=150):
    m,n=np.shape(datamat)
    weights=np.ones(n)
    for j in np.arange(itertimes):
        dataindex=list(np.arange(m))
        for i in np.arange(m):
            alpha=4/(1.0+i+j)+0.01
            randindex=int(np.random.uniform(0,len(dataindex)))
            h=sigmoid(sum(datamat[i]*weights))
            error=labelmat[i]-h
            weights=weights+alpha*error*(np.array(datamat[randindex]))
            dataindex.pop(randindex)
    return weights

def classifyvector(inx,weights):
    prob=sigmoid(sum(inx*weights))
    if prob>0.5: return 1.0
    else: return 0.0
    
def colictest():
    frtrain=open(r'E:\统计学习\机器学习实战数据集\machinelearninginaction\Ch05\horseColicTraining.txt')
    frtest=open(r'E:\统计学习\机器学习实战数据集\machinelearninginaction\Ch05\horseColicTest.txt')
    trainset=[];trainlabels=[]
    for line in frtrain.readlines():
        currline=line.strip().split('\t')
        linearr=[]
        for i in range(21):
            linearr.append(float(currline[i]))
        trainset.append(linearr)
        trainlabels.append(float(currline[21]))
    trainweights=randgradascent1(trainset,trainlabels,500)
    errorcount=0;numtest=0
    for line in frtest.readlines():
        numtest+=1.0
        linearr=[]
        currline=line.strip().split('\t')
        for i in range(21): 
            linearr.append(float(currline[i]))
        if int(classifyvector(linearr,trainweights))!=int(currline[21]):
            errorcount+=1
    errorrate=errorcount/numtest
    print('the error rate of this test is: %f' % errorrate)
    return errorrate
def repeattest():
    errorsum=0;iternum=10
    for i in range(iternum):
        errorsum+=colictest()
    print('after %d iterations,the average error rate is: %f' % (iternum,errorsum/iternum))