# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 08:43:04 2018

@author: 科
"""

import numpy as np
import matplotlib.pyplot as plt
from os import listdir

def loaddata(filename):
    data=np.loadtxt(filename,delimiter='\t')
    datamat=data[:,0:2]#产生的是二维数组
    labelmat=data[:,2 ]#产生的是一维横数组,如果是data[:,2:3]则是二维列数组
    return datamat,labelmat

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def svmSimple(dataArr,labelArr,C,toler,maxIter):
    datamatrix=np.mat(dataArr)
    labelmat=np.mat(labelArr).T
    b=0
    m,n=np.shape(datamatrix)
    alphas=np.mat(np.zeros((m,1)))
    Iter=0
    while Iter<maxIter:
        alphaPairsChanged=0
        for i in range(m):
            fXi=float(np.multiply(alphas,labelmat).T*(datamatrix*datamatrix[i,:].T))+b
            Ei=fXi-float(labelmat[i])
            if ((labelmat[i]*Ei<-toler)and(alphas[i]<C))or((labelmat[i]*Ei>toler))and(alphas[i]>0):
                j=selectJrand(i,m)
                fXj=float(np.multiply(alphas,labelmat).T*(datamatrix*datamatrix[j,:].T))+b
                Ej=fXj-float(labelmat[j])
                alphaiold=alphas[i].copy()
                alphajold=alphas[j].copy()
                if labelmat[i]!=labelmat[j]:
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H: print('L=H'); continue
                eta=2*datamatrix[i,:]*datamatrix[j,:].T-datamatrix[i,:]*datamatrix[i,:].T-datamatrix[j,:]*datamatrix[j,:].T
                if eta>=0: print('eta>=0'); continue
                alphas[j]-=labelmat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if abs(alphas[j]-alphajold<0.00001): print('j not moving enough'); continue
                alphas[i]+=labelmat[i]*labelmat[j]*(alphajold-alphas[j])
                b1=b-Ei-labelmat[i]*(alphas[i]-alphaiold)*datamatrix[i,:]*datamatrix[i,:].T-labelmat[j]*(alphas[j]-alphajold)*datamatrix[i,:]*datamatrix[j,:].T
                b2=b-Ej-labelmat[i]*(alphas[i]-alphaiold)*datamatrix[i,:]*datamatrix[j,:].T-labelmat[j]*(alphas[j]-alphajold)*datamatrix[j,:]*datamatrix[j,:].T
                if (0<alphas[i])and(alphas[i]<C): b=b1
                elif(0<alphas[j])and(alphas[j]<C): b=b2
                else:b=(b1+b2)/2
                alphaPairsChanged+=1
                print('Iter: %d i:%d,pairs changed %d'% (Iter,i,alphaPairsChanged))
        if alphaPairsChanged==0: Iter+=1
        else: Iter=0
        print('Iter number: %d'% Iter)
    return b,alphas
    

def find_SupportVector(datamatin,classlabel,alphas):
    m,n=datamatin.shape
    newdatamat=[]
    newlabelmat=[]
    new_alphas=[]
    for i in range(m):
        if alphas[i]>0:
            newdatamat.append(datamatin[i])
            newlabelmat.append(classlabel[i])
            new_alphas.append(float(alphas[i]))
    newdatamat=np.mat(newdatamat)
    newlabelmat=np.mat(newlabelmat)
    new_alphas=np.mat(new_alphas)
    W=(np.multiply(newlabelmat,new_alphas))*newdatamat
    return W

def calcWs(alphas,dataArr,labelArr):
    X=np.mat(dataArr)
    labelMat=np.mat(labelArr).T
    m,n=np.shape(X)
    w=np.zeros((n,1))
    for i in range(m):
        w+=np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def draw(datamatin,classlabel,alphas,b):
    X=datamatin[:,0]; xmin=X.min()
    xmax=X.max();     Y=datamatin[:,1]
    ymin=Y.min();     ymax=Y.max()
    C=classlabel
    W=find_SupportVector(datamatin,classlabel,alphas)
    x=np.arange(xmin,xmax,0.01);  y=-(W[0,0]*x+float(b))/W[0,1]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(X,Y,c=C);   ax.plot(x,y)
    ax.set_ylim([ymin-3,ymax+3])
    ax.set_xlabel('feature 1');  ax.set_ylabel('feature 2')

class OptStruct_L(object):
    def __init__(self,datamatin,classlabels,C,toler):
        self.X=datamatin
        self.labelmat=classlabels
        self.C=C
        self.tol=toler
        self.m=np.shape(datamatin)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))

class OptStruct(object):
    def __init__(self,datamatin,classlabels,C,toler,kTup):
        self.X=datamatin
        self.labelmat=classlabels
        self.C=C
        self.tol=toler
        self.m=np.shape(datamatin)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def calcEk_L(oS,k):
    fXk=float(np.multiply(oS.labelmat,oS.alphas).T*(oS.X*oS.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelmat[k])
    return Ek

def calcEk(oS,k):
    fXk=float(np.multiply(oS.labelmat,oS.alphas).T*oS.K[:,k])+oS.b
    Ek=fXk-float(oS.labelmat[k])
    return Ek

def selectJ(i,oS,Ei):
    maxK=-1;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=np.nonzero(oS.eCache[:,0].A)[0]
    if len(validEcacheList)>1:
        for k in validEcacheList:
            if k==i:continue        
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if deltaE>maxDeltaE:
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
        return j,Ej

def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]
    

def innerL_L(i,oS):
    Ei=calcEk(oS,i)
    if ((oS.labelmat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))or((oS.labelmat[i]>oS.tol)and(oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaiold=oS.alphas[i].copy();alphajold=oS.alphas[j].copy
        if oS.labelmat[i]!=oS.labelmat[j]:
            L=max(0,oS.alphas[j]-oS.alphas[i]); H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C); H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:print('L=H')
        return 0
        eta=2*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        if eta>=0: print('eta>=0');return 0
        oS.alphas[j]-=oS.labelmat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if abs(oS.alphas[j]-alphajold<0.00001):
            print('j not moving enough');return 0
        oS.alphas[i]+=oS.labelmat[i]*oS.labelmat[j]*(alphajold-oS.alphas[j])
        updateEk(oS,i)
        b1=oS.b-Ei-oS.labelmat[i]*(oS.alphas[i]-alphaiold)*oS.X[i,:]*oS.X[i,:].T-oS.labelmat[j]*(oS.alphas[j]-alphajold)*oS.X[i,:]*oS.X[j,:].T
        b2=oS.b-Ej-oS.labelmat[i]*(oS.alphas[i]-alphaiold)*oS.X[i,:]*oS.X[j,:].T-oS.labelmat[j]*(oS.alphas[j]-alphajold)*oS.X[j,:]*oS.X[j,:].T
        if (0<oS.alphas[i])and(oS.alphas[i]<oS.C): 
            oS.b=b1
        elif (0<oS.alphas[j])and(oS.alphas[j]<oS.C):
            oS.b=b2
        else: oS.b=(b1+b2)/2
        return 1
    else: return 0


def innerL(i,oS):
    Ei=calcEk(oS,i)
    if ((oS.labelmat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))or((oS.labelmat[i]>oS.tol)and(oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaiold=oS.alphas[i].copy();alphajold=oS.alphas[j].copy
        if oS.labelmat[i]!=oS.labelmat[j]:
            L=max(0,oS.alphas[j]-oS.alphas[i]); H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C); H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:print('L=H')
        return 0
        eta=2*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta>=0: print('eta>=0');return 0
        oS.alphas[j]-=oS.labelmat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if abs(oS.alphas[j]-alphajold<0.00001):
            print('j not moving enough');return 0
        oS.alphas[i]+=oS.labelmat[i]*oS.labelmat[j]*(alphajold-oS.alphas[j])
        updateEk(oS,i)
        b1=oS.b-Ei-oS.labelmat[i]*(oS.alphas[i]-alphaiold)*oS.K[i,i]-oS.labelmat[j]*(oS.alphas[j]-alphajold)*oS.K[i,j]
        b2=oS.b-Ej-oS.labelmat[i]*(oS.alphas[i]-alphaiold)*oS.K[i,j]-oS.labelmat[j]*(oS.alphas[j]-alphajold)*oS.K[j,j]
        if (0<oS.alphas[i])and(oS.alphas[i]<oS.C): 
            oS.b=b1
        elif (0<oS.alphas[j])and(oS.alphas[j]<oS.C):
            oS.b=b2
        else: oS.b=(b1+b2)/2
        return 1
    else: return 0
    
    
def smoP(datamatin,classlabels,C,toler,maxIter,kTup=('lin',0)):
    oS=OptStruct(np.mat(datamatin),np.mat(classlabels).T,C,toler,kTup)
    Iter=0
    entireSet=True
    alphaPairsChanged=0
    while (Iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
                print('fullset, Iter: %d i:%d,pairs changed: %d'%(Iter,i,alphaPairsChanged))
            Iter+=1
        else:
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print('non-bound,Iter: %d i: %d,pairs changed %d'%(Iter,i,alphaPairsChanged))
            Iter+=1
        if entireSet:
            entireSet=False
        elif(alphaPairsChanged==0):
            entireSet=True
        print('iteration number: %d'%Iter)
    return oS.b,oS.alphas

def kernelTrans(X,A,kTup):
    m,n=np.shape(X)
    K=np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=np.e**(K/(-1*kTup[1]**2))
    else:
        raise NameError('Jack, we have a serious problem--That kernel is not recognized')
    return K

def testRbf(k1=1.3):
    dataArr,labelArr=loaddata('./machinelearninginaction/Ch06/testSetRBF.txt')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,1000,('rbf',k1))
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).T
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print('There are %d Support vectors'% np.shape(sVs)[0])
    m,n=np.shape(dataMat)
    errorcount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=float(kernelEval.T*np.multiply(labelSV,alphas[svInd]))+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorcount+=1
    print('The training error rate is: %f'% float(errorcount/m))
    
    dataArr,labelArr=loaddata('./machinelearninginaction/Ch06/testSetRBF.txt')
    errorcount=0
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).T
    m,n=np.shape(dataMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=float(kernelEval.T*np.multiply(labelSV,alphas[svInd]))+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorcount+=1
    print('The test error rate is: %f'% float(errorcount/m))

def img2vector(filename):
    returnvector=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        linestr=fr.readline()
        for j in range(32):
            returnvector[0,32*i+j]=linestr[j]
    return returnvector

def loadImages(dirname):
    hwLabels=[]
    trainingFileList=listdir(dirname)
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr==9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:]=img2vector(r'%s/%s'%(dirname,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImages(r'D:\py文件\machinelearninginaction\Ch02\digits\trainingDigits')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).T
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).T
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print('There are %d Support vectors'% np.shape(sVs)[0])
    m,n=np.shape(dataMat)
    errorcount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],kTup)
        predict=float(kernelEval.T*np.multiply(labelSV,alphas[svInd]))+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorcount+=1
    print('The training error rate is: %f'% float(errorcount/m))
    
    dataArr,labelArr=loaddata('./machinelearninginaction/Ch06/testSetRBF.txt')
    errorcount=0
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).T
    m,n=np.shape(dataMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],kTup)
        predict=float(kernelEval.T*np.multiply(labelSV,alphas[svInd]))+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorcount+=1
    print('The test error rate is: %f'% float(errorcount/m))
    
    
dataArr,labelArr=loaddata('./machinelearninginaction/Ch06/testSet.txt')
'''
b,alphas=svmSimple(dataArr,labelArr,0.6,0.001,40)
W=find_SupportVector(dataArr,labelArr,alphas)
draw(dataArr,labelArr,alphas,b)
'''
B,Alphas=smoP(dataArr,labelArr,0.6,0.001,40,kTup=('lin',0))
'''
WW=find_SupportVector(dataArr,labelArr,Alphas)
draw(dataArr,labelArr,Alphas,B)
'''