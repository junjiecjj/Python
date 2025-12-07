import sys
class employee:
    def __init__(self):
        self.num=0
        self.salary=0
        self.name=''
        self.next=None
        
def del_ptr(head,ptr):  #删除节点子程序
    top=head
    if ptr.num==head.num:  #[情形1]:要删除的节点在链表头部
        head=head.num
        print('已删除第 %d 号员工 姓名：%s 薪资:%d' %(ptr.num,ptr.name,ptr.salary))
    else:
        while top.next!=ptr:  #找到删除节点的前一个位置
            top=top.next
        if ptr.next==None:   #删除在链表末尾的节点
            top.next=None
            print('已删除第 %d 号员工 姓名：%s 薪资:%d' %(ptr.num,ptr.name,ptr.salary))
        else:
            top.next=ptr.next #删除在串行中的任一节点
            print('已删除第 %d 号员工 姓名：%s 薪资:%d' %(ptr.num,ptr.name,ptr.salary))
    return head  #返回链表

def main():
    findword=0
    namedata=['Allen','Scott','Marry','John',\
              'Mark','Ricky','Lisa','Jasica',\
              'Hanson','Amy','Bob','Jack']
    data=[[1001,32367],[1002,24388],[1003,27556],[1007,31299], \
          [1012,42660],[1014,25676],[1018,44145],[1043,52182], \
          [1031,32769],[1037,21100],[1041,32196],[1046,25776]]
    print('员工编号 薪水 员工编号 薪水 员工编号 薪水 员工编号 薪水')
    print('-------------------------------------------------------')
    for i in range(3):
        for j in range(4):
            print('%2d  [%3d]  ' %(data[j*3+i][0],data[j*3+i][1]),end='')
        print()
    head=employee() #建立链表头部
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
        newnode.num=data[i][0]
        newnode.next=None
        ptr.next=newnode
        ptr=ptr.next
			
    while(True):
        findword=int(input('请输入要删除的员工编号,要结束删除过程,请输入-1：'))
        if(findword==-1): #循环中断条件
            break
        else:
            ptr=head
            find=0
            while ptr!=None:
                if ptr.num==findword:
                    ptr=del_ptr(head,ptr)
                    find=find+1
                    head=ptr
                ptr=ptr.next
            if find==0:
                print('######没有找到######')
			
    ptr=head
    print('\t员工编号    姓名\t薪水')   #打印剩余链表中的数据
    print('\t==============================')
    while(ptr!=None):
        print('\t[%2d]\t[ %-10s]\t[%3d]' %(ptr.num,ptr.name,ptr.salary))
        ptr=ptr.next
main()
