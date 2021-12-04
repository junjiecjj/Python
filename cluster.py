import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import mixture
import csv
import pandas as pd
import time
from sklearn.metrics import adjusted_rand_score
data=pd.read_csv(r'E:\模式识别\数据(1)\数据\DEAP\EEG_feature.txt')
subject_video=pd.read_csv(r'E:\模式识别\数据(1)\数据\DEAP\subject_video.txt')
valence_arousal_label=pd.read_csv(r'E:\模式识别\数据(1)\数据\DEAP\valence_arousal_label.txt')

print(data.head())
print(data.shape)

print(subject_video.head())
print(subject_video.shape)

print(valence_arousal_label.head())
print(valence_arousal_label.shape)

def test_Kmeans_nclusters(data,label,name):
    X=data
    label_true=label
    nums=range(1,50)
    ARIs=[]
    Distances=[]
    for num in nums:
        clst=cluster.KMeans(n_clusters=num)
        clst.fit(X)
        predicted_labels=clst.predict(X)
        ARIs.append(adjusted_rand_score(label_true,predicted_labels))
        Distances.append(clst.inertia_)

    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs,marker='+')
    ax.set_xlabel('n_cluster')
    ax.set_ylabel('ARI')

    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances,marker='o')
    ax.set_xlabel('n_clusters')
    ax.set_ylabel('intertia_')
    fig.suptitle(name)
    plt.savefig(name)


test_Kmeans_nclusters(data,subject_video['person'],'KMeans-nclusters-person.jpg')
test_Kmeans_nclusters(data,subject_video['video'],'KMeans-nclusters-video.jpg')
test_Kmeans_nclusters(data,valence_arousal_label['happe'],'KMeans-nclusters-happy.jpg')
test_Kmeans_nclusters(data,valence_arousal_label['rouse'],'KMeans-nclusters-arouse.jpg')

def test_Kmeans_n_init(data,label,name):
    
    X=data
    label_true=label
    
    nums=range(1,50)
    ARIs_k=[]
    ARIs_r=[]
    Distance_r=[]
    Distance_k=[]
    
    for num in nums:
        clst=cluster.KMeans(n_init=num,init='k-means++')
        clst.fit(X)
        predicted_labels=clst.predict(X)
        ARIs_k.append(adjusted_rand_score(label_true,predicted_labels))
        Distance_k.append(clst.intertia_)
        
        clst=cluster.KMeans(n_init=num,init='random')
        clst.fit(X)
        predicted_labels=clst.predict(X)
        ARIs_r.append(adjusted_rand_score(label_true,predicted_labels))
        Distance_r.append(clst.intertia_)
        
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs_k,marker='+',label='k-means++')
    ax.plot(nums,ARIs_r,marker='+',label='random')
    ax.set_xlabel("n_init")
    ax.set_ylabel('ARI')
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distance_k,marker='o',label='k-means++')
    ax.plot(nums,Distance_r,marker='0',label='random')
    ax.set_xlabel('n_init')
    ax.set_ylabel('intertia_')
    ax.legend(loc='best')
    fig.suptitle(name)
    plt.savefig(name)
    
test_Kmeans_n_init(data,subject_video['person'],'KMeans-ninit-person.jpg')
test_Kmeans_n_init(data,subject_video['video'],'KMeans-ninit-video.jpg')
test_Kmeans_n_init(data,valence_arousal_label['happe'],'KMeans-ninit-happy.jpg')
test_Kmeans_n_init(data,valence_arousal_label['rouse'],'KMeans-ninits-arouse.jpg')
    


def test_GMM_cov_type(data,label,name):
    X=data
    label_true=label
    nums=range(1,50)
    cov_types=['spherical','tied','diag','full']
    marker='+o*s'
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    
    for i,cov_type in enumerate(cov_types):
        ARIs=[]
        for num in nums:
            clst=mixture.GaussianMixture(n_clusters=num,covariance_type=cov_type)
            clst.fit(X)
            predicted_labels=clst.predict(X)
            ARIs.append(adjusted_rand_score(label_true,predicted_labels))
        ax.plot(nums,ARIs,markers[i],label='covariance_type:%s'%cov_type)

    ax.set_xlabel('n_components')
    ax.set_ylabel('ARI')
    fig.suptitle(name)
    plt.savefig(name)    
    
test_GMM_cov_type(data,subject_video['person'],'GMM-person.jpg')
test_GMM_cov_type(data,subject_video['video'],'GMM-video.jpg')
test_GMM_cov_type(data,valence_arousal_label['happe'],'GMM-happy.jpg')
test_GMM_cov_type(data,valence_arousal_label['rouse'],'GMM-arouse.jpg')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

