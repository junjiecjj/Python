import sys

class employee:
    def __init__(self):
        self.num=0
        self.salary=0
        self.name=''
        self.next=None

def findnode(head,num):
    ptr=head
    while ptr.next!=head:
        if ptr.num==num:
            return ptr
        ptr=ptr.next
    ptr=None   
    return ptr

def deletenode(head,delnode):
    CurNode=employee()
    PreNode=employee()
    TailNode=employee()
    CurNode=None
    PreNode=None
    TailNode=None
    
    if head==None:
        print('[环形链表已经空了]')
        return None
    else:
        if delnode==head: #要删除的节点是链表头
            CurNode=head
            while CurNode.next!=head:
                CurNode=CurNode.next
                #找到最后一个节点并记录下来
                TailNode=CurNode
                #(1)将链表头移到下一个节点
                head=head.next
                #(2)将链表最后一个节点的指针指向新的链表头
                TailNode.next=head
                return head
        else: #要删除的节点不是链表头
            CurNode=head
            while CurNode.next!=delnode:
                CurNode=CurNode.next
            #(1)找到要删除节点的前一个节点并记录下来
            PreNode=CurNode
            #要删除的节点
            CurNode=CurNode.next
            #(2)将要删除节点的前一个指针指向要删除节点的下一个节点
            PreNode.next=CurNode.next
            return head
      
position=0
namedata=['Allen','Scott','Marry','John', \
          'Mark','Ricky','Lisa','Jasica', \
          'Hanson','Amy','Bob','Jack']
data=[[1001,32367],[1002,24388],[1003,27556],[1007,31299], \
          [1012,42660],[1014,25676],[1018,44145],[1043,52182], \
          [1031,32769],[1037,21100],[1041,32196],[1046,25776]]
print('\n员工编号 薪水 员工编号 薪水 员工编号 薪水 员工编号 薪水')
print('-------------------------------------------------------')
for i in range(3):
    for j in range(4):
        print('[%2d] [%3d]  ' %(data[j*3+i][0],data[j*3+i][1]),end='')
    print()
head=employee() #建立链表头
if not head:
    print('Error!! 链表头建立失败!!')
    sys.exit(1)

head.num=data[0][0]
head.name=namedata[0]
head.salary=data[0][1]
head.next=None
ptr=head
for i in range(1,12):  #建立链表
    newnode=employee()
    newnode.num=data[i][0]
    newnode.name=namedata[i]
    newnode.salary=data[i][1]
    newnode.next=None
    ptr.next=newnode #将前一个节点指向新建立的节点
    ptr=newnode #新节点成为前一个节点
	
newnode.next=head #将最后一个节点指向头节点就成了环形链表
while True:
    position=int(input('请输入要删除的员工编号,要结束插入过程,请输入-1：'))
    if position==-1:
        break #循环中断条件
    else:
        ptr=findnode(head,position)
        if ptr==None:
            print('-----------------------')
            print('链表中没这个节点....')
            break
        else:
            head=deletenode(head,ptr)
            print('已删除第 %d 号员工 姓名：%s 薪资:%d' %(ptr.num,ptr.name,ptr.salary))
                
ptr=head #指向链表的头
print('\t员工编号    姓名\t薪水')
print('\t==============================')

while True:
    print('\t[%2d]\t[ %-10s]\t[%3d]' %(ptr.num,ptr.name,ptr.salary))
    ptr=ptr.next #指向下一个节点
    if head==ptr or head==head.next:
        break