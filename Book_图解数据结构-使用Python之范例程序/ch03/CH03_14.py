class employee:
    def __init__(self):
        self.num=0
        self.salary=0
        self.name=''
        self.llink=None  #左指针字段
        self.rlink=None  #右指针字段

def findnode(head,num):
    ptr=head
    while ptr!=None:
        if ptr.num==num:
            return ptr
        ptr=ptr.rlink
    return ptr

def insertnode(head,ptr,num,salary,name):
    newnode=employee()
    newhead=employee()
    newnode.num=num
    newnode.salary=salary
    newnode.name=name
    if head==None: #双向链表是空的
       newhead.num=num
       newhead.salary=salary
       newhead.name=name
       return newhead
    else:
        if ptr==None:
            head.llink=newnode
            newnode.rlink=head
            head=newnode
        else:
            if ptr.rlink==None: #插入链表末尾的位置
                ptr.rlink=newnode
                newnode.llink=ptr
            else:  #插入中间节点的位置
                newnode.rlink=ptr.rlink
                ptr.rlink.llink=newnode
                ptr.rlink=newnode
                newnode.llink=ptr
    return head

llinknode=None
newnode=None
position=0
data=[[1001,32367],[1002,24388],[1003,27556],[1007,31299], \
      [1012,42660],[1014,25676],[1018,44145],[1043,52182], \
      [1031,32769],[1037,21100],[1041,32196],[1046,25776]]
namedata=['Allen','Scott','Marry','John','Mark','Ricky', \
          'Lisa','Jasica','Hanson','Amy','Bob','Jack']

print('员工编号 薪水 员工编号 薪水 员工编号 薪水 员工编号 薪水')
print('-------------------------------------------------------')
for i in range(3):
    for j in range(4):
        print('[%2d] [%3d]  ' %(data[j*3+i][0],data[j*3+i][1]),end='')
    print()

head=employee()  #建立链表头
if head==None:
    print('Error!! 内存分配失败!!')
    sys.exit(0)
else:
    head.num=data[0][0]
    head.name=namedata[0]
    head.salary=data[0][1]
    llinknode=head
    for i in range(1,12):  #建立链表
        newnode=employee()
        newnode.num=data[i][0]
        newnode.name=namedata[i]
        newnode.salary=data[i][1]
        llinknode.rlink=newnode
        newnode.llink=llinknode
        llinknode=newnode

while True:
    print('请输入要插入其后的员工编号,如输入的编号不在此链表中,')
    position=int(input('新输入的员工节点将视为此链表的链表头,要结束插入过程,请输入-1：'))
    if position==-1: #循环中断条件
        break
    else:
        ptr=findnode(head,position)
        new_num=int(input('请输入新插入的员工编号：'))
        new_salary=int(input('请输入新插入的员工薪水：'))
        new_name=input('请输入新插入的员工姓名：')
        head=insertnode(head,ptr,new_num,new_salary,new_name)
       
print('\t员工编号    姓名\t薪水')         
print('\t==============================')
ptr=head
while ptr!=None:
    print('\t[%2d]\t[ %-10s]\t[%3d]' %(ptr.num,ptr.name,ptr.salary))
    ptr=ptr.rlink