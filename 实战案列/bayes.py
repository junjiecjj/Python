# -*- coding: utf-8 -*-
"""
这是《机器学习实战》“朴素贝叶斯”一章
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import numpy.random as nd

def loadDataSet():　　　　#此函数是生成数据
    postingList=[['my','dog','has','flea','problems','help','please'],\
                 ['maybe','not','take','him','to','dog','park','stupid'],\
                 ['my','dalmation','is','so','cute','I','love','him'],\
                 ['stop','posting','stupid','worthless','garbage'],\
                 ['mr','licks','ate','my','steak','how','to','stop','him'],\
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
# postingList的每一行是一篇文档，classVec是每篇文档的种类

def creatVocabList(dataSet):　　　　　#此函数是得到数组中所有不重复单词组成的向量，称为词汇表
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):　　　#返回文档向量，文档向量的每个位置为0或者１，代表该词是否出现在词汇表中。
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else:
            print('the word:%s is not in my Vocabulary!'% word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)  #得到数组的行数
    numWords=len(trainMatrix[0])      #得到数组的列数
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2　　#2
    p1Denom=2    #改为2
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i] #种类为1的所有文档的文档向量的各个对应位置的和
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=np.log(p1Num/p1Denom)
    p0Vect=np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive  #返回两个向量和一个概率值

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)   #计算P(w|c1)P(c1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)  #计算P(w|c2)P(c2)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=creatVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print("testEntry,'classified as: ",classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print("testEntry,'classified as: ",classifyNB(thisDoc,p0V,p1V,pAb))

listOPosts,listClasses=loadDataSet()
myVocabList=creatVocabList(listOPosts)
returnVec=setOfWords2Vec(myVocabList,listOPosts[0])
testingNB()

trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))

p0v,p1v,pAb=trainNB0(trainMat,listClasses)

mySent='This book is the best book on Python or M.L.  I have ever laid eyes upon'
mySent.split()

regEx=re.compile('\\W*')
'''
listOfTokens=regEx.split(mySent)
[tok.lower() for tok in listOfTokens if len(tok)>0]


emailText=open('./machinelearninginaction/Ch04/email/ham/6.txt').read()
listOfTokens=regEx.split(emailText)
'''
def textParse(bigString):
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        wordList=textParse(open('./machinelearninginaction/Ch04/email/spam/%d.txt'\
                                % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('./machinelearninginaction/Ch04/email/ham/%d.txt'\
                                % i,encoding='gb18030',errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=creatVocabList(docList)
    trainingSet=list(range(50))
    testSet=[]
    for i in range(10):
        randIndex=int(np.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is :',float(errorCount)/len(testSet))

spamTest()
