# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:23:39 2018

@author: 科
"""
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import json
import urllib

def loaddata(filename):#读取.txt数据
    origin_data=np.loadtxt(filename,delimiter='\t')
    data_feature=origin_data[:,:-1]
    data_label=origin_data[:,-1]
    return data_feature,data_label

xArr,yArr=loaddata(r'E:\statistic learning\机器学习实战数据集\machinelearninginaction\Ch08\ex0.txt')

def standRegres(xArr,yArr):#利用最小二乘法求w（xTX）-1XTy
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0:
        print('this matrix is singular,cannot do inverse')
        return
    ws=xTx.I*(xMat.T)*yMat
    return ws

def pictureshow(xArr,yArr):#画图
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    ws=standRegres(xArr,yArr)
    yHat=xMat*ws
    fig=plt.figure()
    ax=fig.add_subplot(111)#flatten的作用是将二维数据变为一维。.A的作用是将矩阵转化为np.array
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],marker='*',s=10)
    ax.plot(xMat[:,1],yHat,color='red')
    plt.show()


def lwlr(testPoint,xArr,yArr,k=1):#对某个点进行局部加权线性回归，返回待测点的输出值
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    m=np.shape(xMat)[0]
    weights=np.mat(np.eye((m)))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]#数组与矩阵相减，还是矩阵
        weights[j,j]=np.exp((diffMat*(diffMat.T))/(-2.0*k**2))
    xTx=xMat.T*weights*xMat
    if np.linalg.det(xTx)==0:
        print('this matrix is singular,cannot do inverse')
        return
    ws=xTx.I*(xMat.T)*weights*yMat
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1):#对所有点都进行局部加权线性回归，返回所有点的预测值
    m=np.shape(testArr)[0]
    yHat=np.zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i,:],xArr,yArr,k)
    return yHat

def lwlrshow(xArr,yArr,k=1):
    xMat=np.mat(xArr)
    strInd=xMat[:,1].argsort(0)
    xSort=xMat[strInd][:,0,:] # xMat[strInd]是三维的，所以后面有[:,0,:]
    yHat=lwlrTest(xArr,xArr,yArr,k)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[strInd],c='red')
    ax.scatter(xMat[:,1].flatten().A[0],np.mat(yArr).T.flatten().A[0],s=4,c='blue')
    plt.show()
    
lwlrshow(xArr,yArr,k=0.01)

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

abx,aby=loaddata(r'D:\py文件\machinelearninginaction\Ch08\abalone.txt')
'''
yHat01=lwlrTest(abx[0:99],abx[0:99],aby[0:99],k=0.1)
yHat1=lwlrTest(abx[0:99],abx[0:99],aby[0:99],k=1.0)
yHat10=lwlrTest(abx[0:99],abx[0:99],aby[0:99],k=10)

rssError(aby[0:99],yHat01.T)
rssError(aby[0:99],yHat1.T)
rssError(aby[0:99],yHat10.T)

yHat01=lwlrTest(abx[100:199],abx[0:99],aby[0:99],k=0.1)
rssError(aby[100:199],yHat01.T)
yHat1=lwlrTest(abx[100:199],abx[0:99],aby[0:99],k=1)
rssError(aby[100:199],yHat1.T)
yHat10=lwlrTest(abx[100:199],abx[0:99],aby[0:99],k=10)
rssError(aby[100:199],yHat10.T)
'''

#此函数的功能是计算岭回归系数
def ridgeRegress(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat+lam*np.eye(np.shape(xMat)[1])
    if np.linalg.det(xTx)==0:
        print('this matrix is singular,cannot do inverse')
        return
    ws=xTx*(xMat.T)*yMat
    return ws

#此函数的功能是对每个lamda计算出一个回归系数，并存在矩阵wMat里面，wMat是30×d维的，
#30是自己设定的lamda个数,d是样本特征，也是回归系数的项数
def ridgeTest(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    yMean=np.mean(yMat,0)#对每一列求均值
    yMat=yMat-yMean
    xMean=np.mean(xMat,0)
    xVar=np.std(xMat,0)
    xMat=(xMat-xMean)/xVar
    numTestPts=30
    wMat=np.zeros((numTestPts,np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegress(xMat,yMat,lam=np.exp(i-10))
        wMat[i,:]=ws.T
    return wMat

ridgeWeigths=ridgeTest(abx,aby)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(range(-10,20),ridgeWeigths)
plt.show()
    
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    yMean=np.mean(yMat,0)
    yMat==yMat-yMean
    xMean=np.mean(xMat,0)
    xStd=np.var(xMat,0)
    xMat=(xMat-xMean)/xStd#这里是先去均值再除以标准差,不是除以方差，
    #这里还是除以方差，因为和书上对比，很多地方把这里搞混了
    m,n=np.shape(xMat)#或者调用sklearn.prepocessing.scale(xMat)一步到位
    returnMat=np.zeros((numIt,n))
    ws=np.zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError=np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat
'''
stageWise(abx,aby,eps=0.001,numIt=5000)
'''
def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    sleep(10)
    myAPIstr='AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL='http://www.gooleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json'%(myAPIstr,setNum)
    pg=urllib.request.urlopen(searchURL)
    retDict=json.loads(pg.read())
    for i in range(len(retDict['item'])):
        try:
            currItem=retDict['items'][i]
            if currItem['product']['condition']=='new':
                newFlag=1
            else:
                newFlag=0
            listOfInv=currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice=item['price']
                if sellingPrice>origPrc*0.5:
                    print('%d\t%d\t%d\t%f\t%f'%(yr,numPce,newFlag,origPrc,sellingPrice))
                    retX.append([yr,numPce,newFlag,origPrc])
                    retY.append(sellingPrice)
        except:print('problem with item %d' % i)
        
def setDataCollect(retX,retY):
    searchForSet(retX,retY,8288,2006,800,49.99)
    searchForSet(retX,retY,10030,2002,3096,269.99)
    searchForSet(retX,retY,10179,2007,5195,499.99)
    searchForSet(retX,retY,10181,2007,3428,199.99)
    searchForSet(retX,retY,10189,2008,5922,299.99)
    searchForSet(retX,retY,10196,2009,3263,249.99)

lgx=[]
lgy=[]
setDataCollect(lgx,lgy)
    
def crossValidation(xArr,yArr,numVal=10):
    m=len(yArr)
    indexList=range(m)
    errorMat=np.zeros((numVal,30))
    for i in range(numVal):
        trainX=[]
        trainY=[]
        testX=[]
        testY=[]
        np.random.shuffle(indexList)
        for j in range(m):
            if j<m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat=ridgeTest(trainX,trainY)
        for k in range(30):
            matTestX=np.mat(testX)
            matTrainX=np.mat(trainX)
            meanTrain=np.mean(matTrainX,0)
            varTrain=np.var(matTrainX,0)
            matTestX=(matTestX-meanTrain)/varTrain
            yEst=matTestX*(np.mat(wMat[k,:]).T)+np.mean(trainY)
            errorMat[i,k]=rssError(yEst.T.A,np.array(testY))
    meanErrors=np.mean(errorMat,0)
    minMean=float(min(meanErrors))  # 或者用minMean=meanErrors.min()
    bestWeights=wMat[np.nonzero(meanErrors==minMean)]
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    meanX=np.mean(xMat,0)
    varX=np.var(xMat,0)
    unReg=bestWeights/varX
    print('the best model from Ridge Regression is:\n',unReg)
    print('with constant term:',-1*sum(np.multiply(meanX,unReg))+np.mean(yMat))
    
        
    
    
    
    
    
    
    
    
    
