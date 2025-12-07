class LinkedList:  #声明链表结构
    def __init__(self):
        self.coef=0
        self.exp=0
        self.next=None
def create_link(data): #建立多项式子程序
    for i in range(4):
        newnode=LinkedList()
        if not newnode:
            print("Error!! 内存分配失败!!")
            sys.exit(0)
        if i==0:
            newnode.coef=data[i]
            newnode.exp=3-i
            newnode.next=None
            head=newnode
            ptr=head
        elif data[i]!=0:
            newnode.coef=data[i]
            newnode.exp=3-i
            newnode.next=None
            ptr.next=newnode
            ptr=newnode
    return head

def print_link(head):  #打印多项式子程序
    while head !=None:
        if head.exp==1 and head.coef!=0:  #X^1时不显示指数
            print("%dX + " %(head.coef), end='')
        elif head.exp!=0 and head.coef!=0:
            print("%dX^%d + " %(head.coef,head.exp), end='')
        elif head.coef!=0: #X^0时不显示变量
            print("%d" %(head.coef))
        head=head.next
    print()
        
def sum_link(a,b): #多项式相加子程序
    i=0
    ptr=b
    plus=[None]*4
    while a!=None: #判断多项式1
        if a.exp==b.exp: #指数相等，系数相加
            plus[i]=a.coef+b.coef
            a=a.next
            b=b.next
            i=i+1
        elif b.exp>a.exp: #B指数较大，把系数赋值给C
            plus[i]=b.coef
            b=b.next
            i=i+1
        elif a.exp>b.exp: #A指数较大，把系数赋值给C
            plus[i]=a.coef
            a=a.next
            i=i+1
    return create_link(plus)     #建立相加结果链表C

def main():
    data1=[3,0,4,2]         #多项式A的系数
    data2=[6,8,6,9]         #多项式B的系数
    #c=LinkedList()
    print("原始多项式：\nA=",end='')
    a=create_link(data1)    #建立多项式A
    b=create_link(data2)    #建立多项式B
    print_link(a)           #打印多项式A
    print("B=",end='')
    print_link(b)           #打印多项式B
    print("多项式相加的结果：\nC=",end='') #C为A、B多项式相加的结果
    print_link(sum_link(a,b))         #打印多项式C
    	
main()