# -*- coding: utf-8 -*-
"""
这是《机器学习实战》的“主成分分析”一章
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def loaddataset(filename):
    fr=open(filename)
    stringArr=[line.strip().split() for line in fr.readlines()]
    dataArr=[list(map(float,line)) for line in stringArr]
    return np.mat(dataArr)

def loaddata2(filename):#方法二，np自带的
    origin_data2=np.loadtxt(filename,delimiter='\t')
    return np.mat(origin_data2)

#此处的datamat每行是一个样本，每列是一个特征，m×d
def pca(datamat,topnfeat=9999999):
    meanvals=np.mean(datamat,axis=0)#对每列求均值
    meanremoved=datamat-meanvals
    covmat=np.cov(meanremoved,rowvar=0)#以每列为一个特征求协方差矩阵,d×d
    eigvals,eigvects=np.linalg.eig(np.mat(covmat))
    #eigvals的每个值分别为协方差矩阵covmat的特征值，eigvects的每列分别为与eigvals对应的特征向量
    valssort=np.argsort(eigvals)#对特征值从小到大排序，依次返回最小的特征值的索引，第二小特征值的索引……
    valssorts=valssort[:-(topnfeat+1):-1]#对valssort进行逆序，变为从大到小，取前topfeat个特征值
    redvects=eigvects[:,valssorts]
    lowddatamat=meanremoved*redvects  #lowddatamat是m×k
    reconmat=(lowddatamat*redvects.T)+meanvals#这一步是重构，m×k,k×d=m×d
    return lowddatamat,reconmat

#此pca是调用python内部模块from sklearn.decomposition import PCA完成的，不同于17行的那个
def pca1(datamat):
    pcA=PCA(n_components=10,whiten='True',svd_solver='full')
    newdata=pcA.fit_transform(datamat)
    explian_variance_ratio=pcA.explained_variance_ratio_
    explainvariance=pcA.explained_variance_
    return newdata,explian_variance_ratio,explainvariance

def draw(datamat,reconmat):
    data=datamat.T
    recondata=reconmat.T
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(data[0].flatten().A[0],data[1].flatten().A[0],marker='^',s=10)
    ax.scatter(recondata[0].flatten().A[0],recondata[1].flatten().A[0],marker='o',s=10,c='red')

def replacenanwithmean():
    datamat=loaddataset(r'E:\统计学习\MLiA_SourceCode\machinelearninginaction\Ch13\secom.data',' ')
    numfeat=np.shape(datamat)[1]
    for i in np.arange(numfeat):
        meanval=np.mean(datamat[np.nonzero(~np.isnan(datamat[:,i].A))[0],i])
        datamat[np.nonzero(np.isnan(datamat[:,i].A))[0],i]=meanval
    return datamat
def draw1(eigvals):
    plt.axis([0,20,0,50000000])
    plt.plot(eigvals)
