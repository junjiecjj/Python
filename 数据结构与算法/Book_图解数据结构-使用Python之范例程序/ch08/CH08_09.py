# 基数排序法：从小到大排序
import random

def inputarr(data,size):
    for i in range(size):
        data[i]=random.randint(0,999) #设置 data 值最大为 3 位数

def showdata(data,size):
    for i in range(size):
        print('%5d' %data[i],end='')
    print()

def radix(data,size): 
     n=1 #n为基数，从个位数开始排序
     while n<=100:
        tmp=[[0]*100 for row in range(10)] # 设置暂存数组，[0~9位数][数据个数]，所有内容均为0
        for i in range(size):    # 对比所有数据
            m=(data[i]//n)%10    # m为 n 位数的值，如 36 取十位数(36/10)%10=3
            tmp[m][i]=data[i]    # 把 data[i] 的值暂存在 tmp 中
        k=0
        for i in range(10):
            for j in range(size):
                if tmp[i][j] != 0:    # 因为一开始设置 tmp ={0}，故不为 0 者即为
                    data[k]=tmp[i][j] # data 暂存在 tmp 中的值，把 tmp 中的值放
                    k+=1              # 回 data[ ]里
        print('经过%3d位数排序后：' %n,end='')
        showdata(data,size)
        n=10*n

def main():
    data=[0]*100
    size=int(input('请输入数组大小(100以下)：'))
    print('您输入的原始数据是：')
    inputarr (data,size)
    showdata (data,size)
    radix (data,size)

main()