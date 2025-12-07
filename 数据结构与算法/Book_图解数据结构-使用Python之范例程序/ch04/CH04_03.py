class Node:  #堆栈链结节点的声明
    def __init__(self):
        self.data=0  #堆栈数据的声明
        self.next=None  #堆栈中用来指向下一个节点

top=None

def isEmpty():
    global top
    if(top==None):
        return 1
    else:
        return 0
    
#将指定的数据压入堆栈
def push(data):
    global top
    new_add_node=Node()
    new_add_node.data=data  #将传入的值指定为节点的内容
    new_add_node.next=top   #将新节点指向堆栈的顶端
    top=new_add_node        #新节点成为堆栈的顶端


#从堆栈弹出数据
def pop():
    global top
    if isEmpty():
        print('===目前为空堆栈===')
        return -1
    else:
        ptr=top         #指向堆栈的顶端
        top=top.next    #将堆栈顶端的指针指向下一个节点
        temp=ptr.data   #弹出堆栈的数据
        return temp     #将从堆栈弹出的数据返回给主程序
        
#主程序
while True:
    i=int(input('要压入堆栈,请输入1,要弹出则输入0,停止操作则输入-1: '))
    if i==-1:
        break
    elif i==1:
        value=int(input('请输入元素值:')) 
        push(value)
    elif i==0:
        print('弹出的元素为%d' %pop())
    
print('============================')
while(not isEmpty()): #将数据陆续从顶端弹出
    print('堆栈弹出的顺序为:%d' %pop()) 
print('============================')