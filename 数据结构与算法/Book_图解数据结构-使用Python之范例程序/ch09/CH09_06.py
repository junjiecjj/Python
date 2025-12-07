import random

INDEXBOX=7       #哈希表元素个数
MAXNUM=13        #数据个数

class Node:      #声明链表结构
    def __init__(self,val):
        self.val=val
        self.next=None

global indextable

indextable=[Node]*INDEXBOX  #声明动态数组

def create_table(val):      #建立哈希表子程序
    global indextable
    newnode=Node(val)
    myhash=val%7              #哈希函数除以7取余数
   
    current=indextable[myhash]
    
    if current.next==None:
        indextable[myhash].next=newnode
    else:
        while current.next!=None:
            current=current.next
    current.next=newnode   #将节点加入链表

def print_data(val):       #打印哈希表子程序
    global indextable
    pos=0
    head=indextable[val].next         #起始指针
    print('   %2d：\t' %val,end='')   #索引地址
    while head!=None:
        print('[%2d]-' %head.val,end='')
        pos+=1
        if pos % 8==7:
            print('\t')
        head=head.next
    print()


#主程序
    
data=[0]*MAXNUM
index=[0]*INDEXBOX

for i in range(INDEXBOX):  #清除哈希表
    indextable[i]=Node(-1)

print('原始数据：')
for i in range(MAXNUM):
    data[i]=random.randint(1,30)    #随机数建立原始数据
    print('[%2d] ' %data[i],end='') #并打印出来
    if i%8==7:
        print('\n')

print('\n哈希表：')
for i in range(MAXNUM):
    create_table(data[i])  #建立哈希表

for i in range(INDEXBOX):
    print_data(i)          #打印哈希表
print()