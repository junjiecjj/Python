# -*- coding: utf-8 -*-
"""
这是《机器学习实战》“聚类”一章
"""

import numpy as np
import matplotlib.pyplot as plt
import math
def loaddataset(filename):
    datamat=[]
    fr=open(filename)
    for line in fr.readlines():
        curline=line.strip().split('\t')
        fltline=list(map(float,curline))
        datamat.append(fltline)
    return datamat

def disteclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def randcent(datamat,k):
    n=np.shape(datamat)[1]
    centroids=np.mat(np.zeros((k,n)))
    for j in range(n):
        minj=min(datamat[:,j])
        rangej=float(max(datamat[:,j])-minj)
        centroids[:,j]=np.mat(minj+rangej*np.random.rand(k,1))
    return centroids

def kmeans(datamat,k,distmeas=disteclud,createcent=randcent):
    m=np.shape(datamat)[0]
    clusterassment=np.mat(np.zeros((m,2)))
    centroids=createcent(datamat,k)
    clusterchange=True
    while clusterchange:
        clusterchange=False
        for i in range(m):
            mindist=np.inf;minindex=-1
            for j in range(k):
                distij=distmeas(datamat[i,:],centroids[j,:])
                if distij<mindist:
                    mindist=distij;minindex=j
            if clusterassment[i,0]!=minindex:
                clusterchange=True
            clusterassment[i,:]=minindex,mindist**2
        print(centroids)
        for cent in range(k):
            ptsinclust=datamat[np.nonzero(clusterassment[:,0]==cent)[0]]
            centroids[cent,:]=np.mean(ptsinclust,axis=0)
    return centroids,clusterassment

def bikmeans(datamat,k,distmeas=disteclud):
    m=np.shape(datamat)[0]
    clusterassment=np.mat(np.zeros((m,2)))
    centroid0=np.mean(datamat,axis=0).tolist()[0]
    centlist=[centroid0]
    for j in range(m):
        clusterassment[j,1]=distmeas(np.mat(centroid0),datamat[j,:])**2

    while len(centlist)<k:
        lowsse=np.inf
        for i in range(len(centlist)):
            ptsincurrcluster=datamat[np.nonzero(clusterassment[:,0].A==i)[0],:]
            centroidmat,splitclustass=kmeans(ptsincurrcluster,2,distmeas)
            ssesplit=sum(splitclustass[:,1])
            ssenotsplit=sum(clusterassment[np.nonzero(clusterassment[:,0].A!=i)[0],1])
            print('ssesplit and ssenotsplit:%f,%f'%(ssesplit,ssenotsplit))
            if (ssesplit+ssenotsplit)<lowsse:
                bestcenttosplit=i
                bestnewcents=centroidmat
                bestclustass=splitclustass.copy()
                lowsse=ssesplit+ssenotsplit
        bestclustass[np.nonzero(bestclustass[:,0].A==1)[0],0]=len(centlist)
        bestclustass[np.nonzero(bestclustass[:,0].A==0)[0],0]=bestcenttosplit
        print('the bestcenttosplit is:',bestcenttosplit)
        print('the len of bestclustass is: ',len(bestclustass))
        centlist[bestcenttosplit]=bestnewcents[0,:].tolist()[0]
        centlist.append(bestnewcents[1,:].tolist()[0])
        clusterassment[np.nonzero(clusterassment[:,0].A==bestcenttosplit)[0],:]=bestclustass
    return np.mat(centlist),clusterassment

def bestplot(datamat,centlist,clusterassment,k):
    for i in range(k):
        datax

import urllib
import json
def geograb(staddress,city):
    apistem='http://where.yahooapis.com/geocode?'
    params={}
    params['flags']='J'
    params['appid']='ppp68N8t'
    params['location']='%s %s' % (staddress,city)
    url_params=urllib.parse.urlencode(params)
    yahooApi=apistem+url_params
    print(yahooApi)
    c=urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massplacefind(filename):
    fw=open(r'E:\统计学习\机器学习实战数据集\machinelearninginaction\Ch10\places.txt')
    for line in open(filename).readlines():
        linearr=line.strip().split('\t')
        retdict=geograb(linearr[1],linearr[2])
        if retdict['Resultset']['Error']==0:
            lat=float(retdict['Resultset']['Results'][0]['latitude'])
            lng=float(retdict['Resultset']['Results'][0]['longitude'])
            print('%s\t%f\t%f' % (linearr[0],lat,lng))
            fw.write('%s\t%f\t%f' % (line,lat,lng))
        else:print('error fetching')
        sleep(1)
    fw.close()

def distslc(vecA,vecB):
    a=np.sin(vecA[0,1]*np.pi/180)*np.sin(vecB[0,1]*np.pi/180)
    b=np.cos(vecA[0,1]*np.pi/180)*(np.cos(vecB[0,1]*np.pi/180))*(np.cos(np.pi*(vecB[0,0]-vecA[0,0])/180))
    return np.arccos(a+b)*6371

def clusterclubs(numclust=5):
    datlist=[]
    for line in open(r'E:\统计学习\机器学习实战数据集\machinelearninginaction\Ch10\places.txt').readlines():
        linearr=line.split('\t')
        datlist.append([float(linearr[4]),float(linearr[3])])
    datamat=np.mat(datlist)
    mycentroids,clustassing=bikmeans(datamat,numclust,distmeas=distslc)
    fig=plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scattermarkers=['s','o','^','8','p','d','v','h','>','<']
    axprops=dict(xticks=[],yticks=[])
    ax0=fig.add_axes(rect,label='ax0',**axprops)
    imgp=plt.imread(r'E:\统计学习\机器学习实战数据集\machinelearninginaction\Ch10\Portland.png')
    ax0.imshow(imgp)
    ax1=fig.add_axes(rect,label='ax1',frameon=False)
    for i in range(numclust):
        ptsincurrcluster=datamat[np.nonzero(clustassing[:,0].A==i)[0],:]
        markerstyle=scattermarkers[i % len(scattermarkers)]
        ax1.scatter(ptsincurrcluster[:,0].A, ptsincurrcluster[:,1].A, marker=markerstyle, s=70)
    ax1.scatter(mycentroids[:,0].A, mycentroids[:,1].A, marker='+', s=200)
    plt.show()
