import sys
import random

class student:   #声明链表结构
    def __init__(self):
        self.num=0
        self.score=0
        self.next=None
        
def create_link(data,num): #建立链表子程序
    for i in range(num):
        newnode=student()
        if not newnode:
            print('Error!! 内存配置失败!!')
            sys.exit(0) 
        if i==0:  #建立链表头
            newnode.num=data[i][0]
            newnode.score=data[i][1]
            newnode.next=None
            head=newnode
            ptr=head
        else:  #建立链表的其他节点
            newnode.num=data[i][0]
            newnode.score=data[i][1]
            newnode.next=None
            ptr.next=newnode
            ptr=newnode
        newnode.next=head
    return ptr    #返回链表

def print_link(head): #打印链表子程序
    i=0
    ptr=head.next
    while True:
        print('[%2d-%3d] => ' %(ptr.num,ptr.score),end='\t')
        i=i+1
        if i>=3 : #每行打印三个元素
            print()
            i=0
        ptr=ptr.next
        if ptr==head.next:
            break

def concat(ptr1,ptr2): #连接链表的子程序
    head=ptr1.next  #在ptr1和ptr2中，各找任意一个节点
    ptr1.next=ptr2.next  #把两个节点的next对调即可
    ptr2.next=head
    return ptr2

data1=[[None] * 2 for row in range(6)]
data2=[[None] * 2 for row in range(6)]

for i in range(1,7):
    data1[i-1][0]=i*2-1
    data1[i-1][1]=random.randint(41,100)
    data2[i-1][0]=i*2
    data2[i-1][1]=random.randint(41,100)
	
ptr1=create_link(data1,6)   #建立链表1
ptr2=create_link(data2,6)   #建立链表2
i=0
print('\n原 始 链 表 数 据：')
print('学号 成绩   \t学号 成绩   \t学号 成绩')
print('==========================================')
print('   链表 1 ：')
print_link(ptr1)
print('   链表 2 ：')
print_link(ptr2)
print('==========================================')
print('连接后的链表：')
ptr=concat(ptr1,ptr2)    #连接两个链表
print_link(ptr)