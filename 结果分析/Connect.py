# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import matplotlib.pyplot as plt
import numpy as np



## 链接数据库
db = pymysql.connect(host='202.127.204.29', user='bhguo',passwd='solaris123',\
                     database='east_database',charset='utf8')
#db = pymysql.connect(host='202.127.204.29', user='bhguo',passwd='solaris123',database='east_database',charset='utf8')
# 创建游标
cur = db.cursor()

cur.execute('select * from disruptions')
result = cur.fetchall()
type(result)
df = pd.DataFrame(list(result))

##  对数据库初步处理
df.columns = ['Shot', 'Ip', 't_disrupt', 'Bt', 'Didt', 'Prad', 'Poh', 'Pnbi', 'Picrf', 'Plh', 'Zeff', 'Zaxis', 'Raxis', 'topology', 'Wmhd', 
              'i_halo_I', 'i_halo_u', 'vde', 'direction', 'Tpf', 'timestamp', 'flottop', 'nebar', 'Te0', 'Ti0', 'cause', 'aminor', 'betan', 
              'betap', 'elong', 'li', 'q95', 'gap_outer', 'gap_inner', 't_at_max_halo', 'i_at_max_halo', 'locked_mode', 'z_velocity', 'spike', 
              'spike_time', 'rotation_velocity']  # 1.修改列名
df_new = df.sort_index(axis=0, ascending=True, by='Shot') # 2.df_new是按shot从小到大排序后的
df_new_2014 = df_new[df_new.Shot>=45385]  #选取大于45385的炮，即2014年以后的实验
df_new_2014.index = range(0,len(df_new_2014))  # 重新建立索引index，从0开始
df_new_2014_no_flattop = df_new_2014[df_new_2014.flottop==0] # 非平顶端
df_new_2014_no_flattop.index = range(0,len(df_new_2014_no_flattop))  # 重新建立索引index，从0开始
df_new_2014_flattop = df_new_2014[df_new_2014.flottop==1] # 平顶端
df_new_2014_flattop.index = range(0,len(df_new_2014_flattop))  # 重新建立索引index，从0开始



"""

x1 = np.arange(0, len(df_new_2014_flattop)) # 建立一个与处理后数据库炮号量相等的一位数组
x2 = np.arange(0, len(df_new_2014_no_flattop))
y1 = np.array(df_new_2014_flattop['betan']) # 将某一列数据类型变成数组
y2 = np.array(df_new_2014_no_flattop['betan']) 
plt.scatter(x1, y1, c='red', alpha=1, marker='.', label='betan') # 画x，y的点图。c是color颜色，alpha是透明度，marker是画的样式
plt.scatter(x2, y2, c='blue', alpha=1, marker='^', label='brtan') 



"""