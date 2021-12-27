# -*- coding: utf-8 -*-
"""
这是《机器学习实战》提升算法一章
"""

import numpy as np
import matplotlib.pyplot as plt

'''机器学习实战里面的数据的标签都是m×n维的，m是样本个数，n是每个样本的特征'''
def loadSimpData():
    datMat=np.matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

dataMat,classLabels=loadSimpData()

#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],\
           #marker=(9,3,30),c=np.array(classLabels))

'''选择某一列特征作为分类的对象，根据threshIneq是lt还是gt使当特征大于或小于某一值时分为+1，'''
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

'''
此函数的功能是找出具有多维特征的样本的最合适的特征i是哪维，最合适的分类阈值是多少threshVal，
以及最合适的左边还是右边(lt,gt),而找出这些值的标准是分类的错误率最小，这里的错误率的计算不是错误样本
除以总样本，而是错分样本的权值乘以1；返回字典形式的上述的三个值，最小的错误率是多少，以及错误率最小时
的分类标签结果
'''
def buildStump(dataArr,classLabels,D):
    dataMatrix=np.mat(dataArr)
    labelMat=np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps=10.0
    bestStump={}
    bestClasEst=np.mat(np.zeros((m,1)))
    minError=np.inf
    for i in range(n):
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps+1)):
            for inequal in['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=np.mat(np.ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr
               # print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=9):
    weakClassArr=[]
    m=np.shape(dataArr)[0]
    D=np.mat(np.ones((m,1))/m)
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
      #  print('D: ',D.T)
        alpha=float(0.5*np.log((1-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
       # print('classEst: ',classEst)
        expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D=np.multiply(D,np.exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
       # print('aggClassEst: ',aggClassEst.T)
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        #布尔值可以和数值直接相乘，相当于False=0,True=1
        errorRate=aggErrors.sum()/m
        print('Total error rate: %.3f \n' %  errorRate)
        if errorRate==0:
            break
    return weakClassArr,aggClassEst

# classifierArray=adaBoostTrainDS(dataMat,classLabels,numIt=9)

def adaClassify(datToClass,classifierArr):
    dataMatrix=np.mat(datToClass)
    m=np.shape(dataMatrix)[0]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                               classifierArr[i]['thresh'],\
                               classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        #print(aggClassEst)
    return np.sign(aggClassEst)

def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
'''
datArr,labelArr=loadDataSet('/home/jack/公共的/py文件/machinelearninginaction/Ch07/horseColicTraining2.txt')
classifierArray,aggClassEst=adaBoostTrainDS(datArr,labelArr,10)

testArr,testLabelArr=loadDataSet('/home/jack/公共的/py文件/machinelearninginaction/Ch07/horseColicTest2.txt')
prediction10=adaClassify(testArr,classifierArray)

errArr=np.mat(np.ones((67,1)))
a=errArr[prediction10!=np.mat(testLabelArr).T].sum()
print(a)
'''

def plotROC(preStrengths,classLabels):
    cur=(1.0,1.0)
    ySum=0.0
    numPosClass=sum(np.array(classLabels)==1.0)
    yStep=1/float(numPosClass)
    xStep=1/float(len(classLabels)-numPosClass)
    sortedIndicies=preStrengths.argsort()
    fig=plt.figure()
    fig.clf() #有没有这行代码不影响
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0
            delY=yStep
        else:
            delX=xStep
            delY=0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print('The Area Under the curve is : ',ySum*xStep)

datArr,labelArr=loadDataSet('/home/jack/公共的/py文件/machinelearninginaction/Ch07/horseColicTraining2.txt')
classifierArray,aggClassEst=adaBoostTrainDS(datArr,labelArr,1000)#10,100,200,500,1000，AUC曲线的面积逐渐增大。
plotROC(aggClassEst.T,labelArr)
