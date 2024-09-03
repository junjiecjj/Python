#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:52:04 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODY4ODE5MQ==&mid=2247484314&idx=1&sn=c4dc11f14cabc26c679c111d7d1fbb36&chksm=ce87bc8a445a5d12a6a3ec609da41e9fed9ab52e60d4d7620efe1fa19f2b1f5c61406f5cb702&mpshare=1&scene=1&srcid=0901i0vww5j4pgW3Lz3a5MD5&sharer_shareinfo=dbbf6a7d72b7c7f85cc0be05e61cafeb&sharer_shareinfo_first=dbbf6a7d72b7c7f85cc0be05e61cafeb&exportkey=n_ChQIAhIQ5UPKg84lxz27g3RtRvdi3RKfAgIE97dBBAEAAAAAAOeZAMP0txAAAAAOpnltbLcz9gKNyK89dVj02wTRFLuQe0Lbz7OcGf7AkIqLM4O0l03wb6LsTU9L7r8eoFCmY3hjpG2QoXn7On8EHqawHgFsaETCnE4HZm8JkYN79JNh0%2F1tTQa0jM3F2zQKP6XGIG2ZdUxT47pj2a5nwdLBpyDaS4sexyt1cpce6XRV0ON3kUB1nsom9HFKfST0ldAp%2BkXFA7gAE6MR0HWt4D6Ilr54m0XznmeQol4g2TQ0zyBS9HjCApRNggR8yYGtn9yYWyFUqP2NztfOH1z%2Bnklm%2FCyReOTWpVci7wHDfwIbJhizvIg96FpbPLB3ktU5Quu7vo%2FsYScsIy11efrWZ31oLv69DiIY&acctmode=0&pass_ticket=%2Fj%2F9xFtz71t7HUPS0jJ8XktFolSnuSWDWHHJhQ2GpYdL35g8ph%2BscPEK8CFgpovX&wx_header=0#rd


"""


import random
import numpy as np
import math

num_city=30#城市总数0-29
num_total=100#随机生成的初始解的总数
copy_num=70#保留的解的个数
cross_num=20#交叉解的个数
var_num=10#变异解的个数

location=np.loadtxt('city_location.txt')
#print(location)

#随机生成初始解[[],[],[]...]
def generate_initial():
    initial=[]
    city=list(range(num_city))
    for i in range(num_total):
        random.shuffle(city)
        p=city.copy()
        while (p in initial):
            #print('2333')#随机了一个重复的解
            random.shuffle(city)
            p=city.copy()
        initial.append(p)
    return initial


#对称矩阵，两个城市之间的距离
def distance_p2p_mat():
    dis_mat=[]
    for i in range(30):
        dis_mat_each=[]
        for j in range(30):
            dis=math.sqrt(pow(location[i][0]-location[j][0],2)+pow(location[i][1]-location[j][1],2))
            dis_mat_each.append(dis)
        dis_mat.append(dis_mat_each)
   # print(dis_mat)
    return dis_mat


#目标函数计算,适应度计算，中间计算。适应度为1/总距离*10000
def dis_adp_total(dis_mat,initial):
    dis_adp=[]
#    dis_test=[]
    for i in range(num_total):
        dis=0
        for j in range(num_city-1):
            dis=dis_mat[initial[i][j]][initial[i][j+1]]+dis
        dis=dis_mat[initial[i][29]][initial[i][0]]+dis#回家
#        dis_test.append(dis)
        dis_adp_each= 10000.0/dis
        dis_adp.append(dis_adp_each)
#    print(dis_test)
    return dis_adp




def choose_fromlast(dis_adp,answer_source):
    mid_adp=[]
    mid_adp_each=0
    for i in range(num_total):
        mid_adp_each=dis_adp[i]+mid_adp_each
        mid_adp.append(mid_adp_each)
   # print(mid_adp)
    #产生0-mid_adp[num_total-1]之间的随机数
    #选择n-1<随机数<n的那个n的解,保留
    copy_ans=[]
    for p in range(copy_num):
        rand=random.uniform(0,mid_adp[num_total-1])#产生随机数
       # print(rand)
       # print(p)
        for j in range(num_total):
            if (rand<mid_adp[j]):#查找位置
                copy_ans.append(answer_source[j])
                break
            else:
                continue
    return copy_ans




#随机选择保留下来的70中的25个进行交叉
def cross_pronew(copy_ans):
    for i in range(cross_num):
        which=random.randint(0,copy_num-1)#选择对那个解交叉
        cross_list=copy_ans[which].copy()
        while (cross_list in copy_ans):
            p=random.randint(0,num_city-1)
            q=random.randint(0,num_city-1)
            cross_list[p],cross_list[q]=cross_list[q],cross_list[p]#第一次交换位置
            m=random.randint(0,num_city-1)
            n=random.randint(0,num_city-1)
            cross_list[m],cross_list[n]=cross_list[n],cross_list[m]#第二次交换位置
        copy_ans.append(cross_list)
    cross_ans=copy_ans.copy()
    return cross_ans



#随机选择那95中的5个进行变异
def var_pronew(cross_ans):
    for i in range(var_num):
        which=random.randint(0,copy_num+cross_num-1)#选择对那个解交叉
        var_list=cross_ans[which].copy()
        while (var_list in cross_ans):
            p=random.randint(0,num_city-1)
            q=random.randint(0,num_city-1)
            var_list[p],var_list[q]=var_list[q],var_list[p]#交换位置
        cross_ans.append(var_list)
    var_ans=cross_ans.copy()
    return var_ans

answer_source=generate_initial()
dis_mat=distance_p2p_mat()
#print(dis_mat)
dis_adp=dis_adp_total(dis_mat,answer_source)
adp_max_new=max(dis_adp)
if (max(dis_adp)>10000/700):
    print('找到的最近距离是：',max(dis_adp))

else:
    print('哎呀没找到，我再找找~')
    answer_new=answer_source
    dis_adp_new=dis_adp
    while(adp_max_new<=10000/700):
        copy_answer=choose_fromlast(dis_adp_new,answer_new)
        cross_answer=cross_pronew(copy_answer)
        var_answer=var_pronew(cross_answer)
        answer_new=var_answer.copy()
        dis_adp_new=dis_adp_total(dis_mat,answer_new)
        adp_max_new=max(dis_adp_new)
#        dis_min=10000/adp_max_new
#        print('这次是：',dis_min)


    dis_min=10000/adp_max_new
    print('终于找到你啦：',dis_min)





















