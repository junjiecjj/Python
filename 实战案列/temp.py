# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import xlrd
from scipy import interpolate
from os import listdir


def one_resample(*args):
    for arg in args:
        if arg in(pcr,dfsdev,ic,lmsz):
            pass
        else:
            exec("'f_%s=interpolate.interp1d(arg[1],arg[0],kind='nearest')" % 'arg')
            

def all_resample(filename):
    all_file=listdir(r'D:\disruption-code\数据\密度极限')
    m=len(all_file)
    sheet=xlrd.open_workbook('D:\disruption-code\excel_of_data\密度极限.xlsx')
    table=sheet.sheet_by_name('Sheet1')
    #in:  sheet1.row_values(0)
    #out: [65001.0, 3.321, 'DFSDEV']
    #in:  sheet1.row_values(0)[0]
    #out: 65001.0
    # row_num,col_num=table.nrows,table.ncols
    # table.row(10)[0]   读取第11行第一列的数据
    for i in range(m):
        try:
            pcr=np.loadtxt(r'D:\disruption-code\数据\密度极限\%d\%d_PCR.txt' % (table.row_values(i)[0],table.row_values(i)[0]))
            dfsdev=np.loadtxt(r'D:\disruption-code\数据\密度极限\%d\%d_DFSDEV.txt' % (table.row_values(i)[0],table.row_values(i)[0]))
            li=np.loadtxt(r'D:\disruption-code\数据\密度极限\%d\%d_LI.txt' % (table.row_values(i)[0],table.row_values(i)[0]))
            sxr=np.loadtxt(r'D:\disruption-code\数据\密度极限\%d\%d_SXR.txt' % (table.row_values(i)[0],table.row_values(i)[0]))
            
        except FileNotFoundError as e:
            continue
        data_Af_resample=one_resample(pcr,dfsdev,li,sxr)
