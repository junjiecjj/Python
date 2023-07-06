from os import listdir
import numpy as np
import operator
def creatdataset():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inx,dataset,labels,k):
    datasetsize=dataset.shape[0]
    diffmat=np.tile(inx,(datasetsize,1))-dataset
    sqdiffmat=diffmat**2
    sqdistances=np.sum(sqdiffmat,axis=1)
    distances=sqdistances**0.5
    sortdistance=np.argsort(distances)
    classcount={}
    for i in range(k):
       voteilabel=labels[sortdistance[i]]
       classcount[voteilabel]=classcount.get(voteilabel,0)+1
    sortedclasscount=sorted(classcount.items(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

def file2matrix(filename):
    fr=open(filename)
    arrayolines=fr.readlines()
    numberoflines=len(arrayolines)
    returnmat=np.zeros((numberoflines,3))
    classlabervector=[]
    index=0
    for line in arrayolines:
       line=line.strip()
       linefromline=line.split('\t')
       returnmat[index,:]=linefromline[0:3]
       classlabervector.append(int(linefromline[3]))
       index=index+1
    return returnmat,classlabervector#注意return必须与for对齐，否则输出的classlabelvector只有一个元素

def autonorm(dataset):
    minvals=dataset.min(0)
    maxvals=dataset.max(0)
    range=maxvals-minvals
    m=dataset.shape[0]
    dataset=dataset-np.tile(minvals,(m,1))
    normdata=dataset/np.tile(range,(m,1))
    return normdata,range,minvals

def datingclasstest():
    horatio=0.1
    datingdatamat,datinglabels=file2matrix(r'F:\datingTestSet2.txt')
    normdata,range,minvals=autonorm(datingdatamat)
    m=normdata.shape[0]
    testnum=int(m*horatio)
    count=0
    for i in np.arange(testnum):
        classifyresult=classify0(normdata[i,:],normdata[testnum:m,:],datinglabels[testnum:m],3)
        print('the classifier result is:%d,the real answer is:%d'%(classifyresult,datinglabels[i]))
        if (classifyresult!=datinglabels[i]):count+=1
    print('the total error rate is:%f'%(count/float(testnum)))
    
def classifyperson():
    resultlist=['not at all','in small doses','in large doses']
    percenttats=float(input('percentage of time spent in playing video games?:'))
    ffmiles=float(input('frequent flier miles earned per year?:'))
    icecream=float(input('how may icecream comsumed each year?:'))
    testdata=np.array([percenttats,ffmiles,icecream])
    datingdatamat,datinglabels=file2matrix(r'F:\datingTestSet2.txt')
    normdata,range,minvals=autonorm(datingdatamat)
    normtestdata=(testdata-minvals)/range
    classifyresult=classify0(normtestdata,datingdatamat,datinglabels,3)
    print('you will probably like this preson:',resultlist[classifyresult-1])
def img2vector(filename):
    returnvector=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        linestr=fr.readline()
        for j in range(32):
            returnvector[0,32*i+j]=linestr[j]
    return returnvector
def handwritingclasstest():
    hwlabels=[]
    trainingfilelist=listdir(r'E:\统计学习\机器学习实战数据集\machinelearninginaction\Ch02\digits\trainingDigits')
    m=len(trainingfilelist)
    trainmat=np.zeros((m,1024))
    for i in range(m):
        filenamestr=trainingfilelist[i].split('.')[0]
        classnumstr=int(filenamestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainmat[i,:]=img2vector('E:\统计学习\机器学习实战数据集\machinelearninginaction\Ch02\digits\trainingDigits\\%s'%(trainingfilelist[i])
    testfilelist=listdir(r'E:\统计学习\机器学习实战数据集\machinelearninginaction\Ch02\digits\testDigits')
    errcount=0
    mtest=len(testfilelist)
    for i in range(mtest):
        filenamestr=testfilelist[i].split('.')[0]
        classnumstr=int(filenamestr.split('_')[0])
        vectorbetest=img2vector('E:\统计学习\机器学习实战数据集\machinelearninginaction\Ch02\digits\testDigits/%s'%testfilelist[i])
        classifyresult=classify0(vectorbetest,trainmat,hwlabels,3)
        if (classifyresult!=classnumstr): errcount+=1
        print('the test result is:%d,the real answer is:%d'%(classifyresult,classnumstr))
        print('the total error number is:%d'%errcount)
        print('the total error rate is:%f'%(errcount/float(mtest)))
