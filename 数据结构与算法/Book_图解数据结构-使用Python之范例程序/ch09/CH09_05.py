import random

INDEXBOX=10    #哈希表最大元素
MAXNUM=7       #最大数据个数

def print_data(data,max_number):  #打印数组子程序
    print('\t',end='')
    for i in range(max_number):
        print('[%2d] ' %data[i],end='')
    print()

def create_table(num,index):  #建立哈希表子程序
    tmp=num%INDEXBOX          #哈希函数 = 数据%INDEXBOX
    while True:
        if index[tmp]==-1:    #如果数据对应的位置是空的
            index[tmp]=num    #则直接存入数据
            break
        else:
            tmp=(tmp+1)%INDEXBOX    #否则往后找位置存放
            
#主程序
index=[None]*INDEXBOX
data=[None]*MAXNUM

print('原始数组值：')
for i in range(MAXNUM):  #起始数据值
    data[i]=random.randint(1,20)
for i in range(INDEXBOX): #清除哈希表
    index[i]=-1
print_data(data,MAXNUM)   #打印起始数据

print('哈希表的内容：')
for i in range(MAXNUM):  #建立哈希表
    create_table(data[i],index)
    print('  %2d =>' %data[i],end='')  #打印单个元素的哈希表位置
    print_data(index,INDEXBOX)

print('完成的哈希表：')
print_data(index,INDEXBOX)  #打印最后完成的结果