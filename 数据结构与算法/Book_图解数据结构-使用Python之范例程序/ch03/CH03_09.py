class employee:
    def __init__(self):
        self.num=0
        self.salary=0
        self.name=''
        self.next=None
        
def findnode(head, num):
    ptr=head
    while ptr.next !=head:
        if ptr.num==num:
            return ptr
        ptr=ptr.next
    return ptr
    
def insertnode(head,after,num,salary,name):
    InsertNode=employee()
    CurNode=None
    InsertNode.num=num
    InsertNode.salary=salary
    InsertNode.name=name
    InsertNode.next=None
    if InsertNode==None:
        print('内存分配失败')
        return None
    else:
        if head==None: #链表是空的
            head=InsertNode
            InsertNode.next=head
            return head
        else:
            if after.next==head: #新增节点在链表头的位置
                #(1)将新增节点的指针指向链表头
                InsertNode.next=head
                CurNode=head
                while CurNode.next!=head:
                    CurNode=CurNode.next
                #(2)找到链表末尾后，将它的指针指向新增节点
                CurNode.next=InsertNode
                #(3)将链表头的指针指向新增节点
                head=InsertNode
                return head
            else: #新增节点在链表头以外的地方
                #(1)将新增节点的指针指向after的下一个节点
                InsertNode.next=after.next
                #(2)将节点after的指针指向新增节点
                after.next=InsertNode
                return head
     
position=0
namedata=['Allen','Scott','Marry','John','Mark','Ricky', \
          'Lisa','Jasica','Hanson','Amy','Bob','Jack']
data=[[1001,32367],[1002,24388],[1003,27556],[1007,31299], \
    [1012,42660],[1014,25676],[1018,44145],[1043,52182], \
    [1031,32769],[1037,21100],[1041,32196],[1046,25776]]

print('员工编号 薪水 员工编号 薪水 员工编号 薪水 员工编号 薪水')
print('-------------------------------------------------------')
for i in range(3):
    for j in range(4):
        print('[%2d] [%3d]  ' %(data[j*3+i][0],data[j*3+i][1]),end='')
    print()

head=employee() #建立链表头
if not head:
    print('Error!! 内存分配失败!!')
    sys.exit(0)

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
    
newnode.next=head#将最后一个节点指向头节点就成了环形链表

while True:
    print('请输入要插入其后的员工编号,如果输入的编号不在此链表中,')
    position=int(input('则新输入的员工节点将视为此链表的第一个节点,要结束插入过程,请输入-1：'))
    if position == -1:  #循环中断条件
        break
    else:
        ptr=findnode(head,position)
        new_num=int(input('请输入新插入的员工编号：'))
        new_salary=int(input('请输入新插入的员工薪水：'))
        new_name=input('请输入新插入的员工姓名：')
        head=insertnode(head,ptr,new_num,new_salary,new_name)
                 
ptr=head #指向链表的头
print('\t员工编号    姓名\t薪水')         
print('\t==============================')

while True:
    print('\t[%2d]\t[ %-10s]\t[%3d]' %(ptr.num,ptr.name,ptr.salary))
    ptr=ptr.next#指向下一个节点
    if head ==ptr or head==head.next:
        break
	
