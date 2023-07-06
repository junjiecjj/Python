# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:38:36 2017

@author: 科
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
#读取数据，每行为一个样本，每列为样本的特征向量，返回一个矩阵，
def loaddataset(filename):
    fr=open(filename)
    datamat=[]
    for line in fr.readlines():
        line=line.strip().split('\t')
        linearr=map(float,line)
        datamat.append(list(linearr))
    return np.array(datamat)
datamat1=loaddataset(r'E:\模式识别\数据(1)\数据\DEAP\EEG_feature.txt')
subject_vedio_label1=loaddataset(r'E:\模式识别\数据(1)\数据\DEAP\subject_video.txt')
valence_arouse_label1=loaddataset(r'E:\模式识别\数据(1)\数据\DEAP\valence_arousal_label.txt')

datamat2=loaddataset(r'E:\模式识别\数据(1)\数据\MAHNOB-HCI\EEG_feature.txt')
subject_vedio_label2=loaddataset(r'E:\模式识别\数据(1)\数据\MAHNOB-HCI\subject_video.txt')
valence_arouse_label2=loaddataset(r'E:\模式识别\数据(1)\数据\MAHNOB-HCI\valence_arousal_label.txt')
EEG_emotion_category2=loaddataset(r'E:\模式识别\数据(1)\数据\MAHNOB-HCI\EEG_emotion_category.txt')
#去中心化，（x-均值）/方差
def zscore(datamat):
    dataarr=preprocessing.scale(datamat)
    return np.mat(dataarr)

#主成分分析
def pca(datamat):
    pcA=PCA(n_components=3,whiten='True',svd_solver='full')
    newdata=pcA.fit_transform(datamat)
    explian_variance_ratio=pcA.explained_variance_ratio_
    explainvariance=pcA.explained_variance_
    return newdata,explian_variance_ratio,explainvariance



def test_kMeans_clusternum(train_datamat,real_label,name):
    X=train_datamat

    clusternum=range(1,60)
    AIRs=[]
    Distance=[]
    for num in clusternum:
        estimator=cluster.KMeans(n_clusters=num)
        estimator.fit(X)
        predict_label=estimator.predict(X)
        AIRs.append(adjusted_rand_score(real_label,predict_label))
        Distance.append(estimator.inertia_)
        
    fig=plt.figure()
    ax=fig.add_subplot(121)
    ax.plot(clusternum,AIRs,marker='+',c='b')
    ax.set_xlabel('clusternum')
    ax.set_ylabel('AIRs')
    
    ax=fig.add_subplot(122)
    ax.plot(clusternum,Distance,marker='*',c='r')
    ax.set_xlabel('clusternum')
    ax.set_ylabel('inertia')
    fig.suptitle(name)
    plt.savefig(name)
    
test_kMeans_clusternum(datamat1,subject_vedio_label1[:,0],'KMeans-ncluster-person.jpg')
test_kMeans_clusternum(datamat1,subject_vedio_label1[:,1],'KMeans-ncluster-vedio.jpg')
test_kMeans_clusternum(datamat1,valence_arouse_label1[:,0],'KMeans-ncluster-happy')
test_kMeans_clusternum(datamat1,valence_arouse_label1[:,1],'KMeans-nclustr-arouse')

def test_KMeans_init(train_datamat,real_label,name):
    X=train_datamat

    nums=range(1,60)
    AIRs_k=[]
    AIRs_r=[]
    Distance_k=[]
    Distance_r=[]
    for num in nums:
        estimator_k=cluster.KMeans(n_init=num,init='k-means++')
        estimator_k.fit(X)
        predict_label=estimator_k.predict(X)
        AIRs_k.append(adjusted_rand_score(real_label,predict_label))
        Distance_k.append(estimator_k.inertia_)
        
        estimator_r=cluster.KMeans(n_init=num,init='random')
        estimator_r.fit(X)
        predict_label=estimator_r.predict(X)
        AIRs_r.append(adjusted_rand_score(real_label,predict_label))
        Distance_r.append(estimator_r.inertia_)
        
    fig=plt.figure()
    ax=fig.add_subplot(121)
    ax.plot(nums,AIRs_k,'r+',label='k-means++')
    ax.plot(nums,AIRs_r,'b-',label='random')
    ax.set_xlabel('n_init')
    ax.set_ylabel('AIRs')
    ax.set_ylim(-1,1)
    ax.legend(loc='best')
    
    ax=fig.add_subplot(122)
    ax.plot(nums,Distance_k,'g+',label='k-means++')
    ax.plot(nums,Distance_k,'k-',label='random')
    ax.set_xlabel('clusternum')
    ax.set_ylabel('inertia')
    ax.legend(loc='best')
    
    fig.suptitle(name)
    plt.savefig(name)
      
test_KMeans_init(datamat1,subject_vedio_label1[:,0],'KMeans-n_init-person.jpg')
test_KMeans_init(datamat1,subject_vedio_label1[:,1],'KMeans-n_init-vedio.jpg')
test_KMeans_init(datamat1,valence_arouse_label1[:,0],'KMeans-n-init-happy')
test_KMeans_init(datamat1,valence_arouse_label1[:,1],'KMeans-n_init-arouse')



def test_hierarchicalcluster_num_link(train_datamat,real_label,name):
    X=train_datamat
    clusternum=range(1,60)
    linkage_type=['ward', 'complete', 'average']
    markers='+s*'
    color='rbg'
    fig=plt.figure()
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    for i,aff in enumerate(linkage_type):
        AIRs=[]
        AMIs=[]
        
        for num in clusternum:
            estimator=AgglomerativeClustering(n_clusters=num,linkage=aff)
            estimator.fit(X)
            predict_label=estimator.fit_predict(X)
            AIRs.append(adjusted_rand_score(real_label,predict_label))
            AMIs.append(adjusted_mutual_info_score(real_label,predict_label))
                    
        ax1.plot(clusternum,AIRs,marker=markers[i],c=color[i],label='linkage:%s'%aff)
        ax1.set_xlabel('clusternum')
        ax1.set_ylabel('AIRs')
        ax1.legend(loc='upper right')
        
        ax2.plot(clusternum,AMIs,marker=markers[i],c=color[i],label='linkage:%s'%aff)
        ax2.set_xlabel('clusternum')
        ax2.set_ylabel('AMIs')
        ax2.legend(loc='upper right')
        
    fig.suptitle(name)
    plt.savefig(name)
    plt.show()
        
        
test_hierarchicalcluster_num_link(datamat1,subject_vedio_label1[:,0],'hierarchicalcluster_linkage_clusternum_person.jpg')
test_hierarchicalcluster_num_link(datamat1,subject_vedio_label1[:,1],'hierarchicalcluster_linkage_clusternum_vedio.jpg')
test_hierarchicalcluster_num_link(datamat1,valence_arouse_label1[:,0],'hierarchicalcluster_linkage_clusternum_happy.jpg')
test_hierarchicalcluster_num_link(datamat1,valence_arouse_label1[:,1],'hierarchicalcluster_linkage_clusternum_arouse.jpg')

def pLot2D(datamat,label,classnumber,centroid=None):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    scattermarkers=['s','o','^','8','p','d','v','h','>','<']
    for i in range(classnumber):
        classdata=datamat[np.nonzero(label[:]==i)[0],:]
        markerstyle=scattermarkers[i % len(scattermarkers)]
        ax.scatter(classdata[:,0],classdata[:,1],marker=markerstyle,s=10)
    if centroid==None:
        pass
    else:
        ax.scatter(centroid[:,0],centroid[:,1],marker='+',s=200)
    ax.set_xlabel('attribute 1')
    ax.set_ylabel('attribute 2')

def plot3D(datamat,label,classnumber,centroid=None):
    fig1=plt.figure()
    ax=Axes3D(fig1)
    scattermarkers=['s','o','^','8','p','d','v','h','>','<']
    for i in range(classnumber):
        classdata=datamat[np.nonzero(label[:]==i)[0],:]
        markerstyle=scattermarkers[i % len(scattermarkers)]
        ax.scatter(classdata[:,0],classdata[:,1],classdata[:,2],marker=markerstyle,s=10)
    if centroid==None:
        pass
    else:
        ax.scatter(centroid[:,0],centroid[:,1],centroid[:,2],marker='+',s=500)
    
def accurate(label):
    fl=open(r'E:\模式识别\数据(1)\数据\MAHNOB-HCI\EEG_emotion_category.txt')
    reallabel=[]
    errorcount=0
    for line in fl.readlines():
        linearr=line.strip().split('\t')
        line=list(map(int,linearr))
        reallabel.append(line[0])
    rlabel=np.array(reallabel)
    crlabel=rlabel.copy()
    for i in [11,12]:
        crlabel[np.nonzero(crlabel[:]==i)[0]]=i-4
    m=label.shape[0]
    for i in range(m):
        if label[i]!=reallabel[i]:
            errorcount+=1
    errorrate=errorcount/m
    print('the error rate of cluster is: ',errorrate)
    return errorrate,rlabel,crlabel
