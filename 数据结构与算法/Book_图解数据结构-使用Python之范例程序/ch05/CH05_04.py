class Node:
    def __init__(self):
        self.data=0
        self.next=None
        
front=Node()
rear=Node()
front=None
rear=None

#方法enqueue:队列数据的加入
def enqueue(value):
    global front
    global rear
    node=Node()  #建立节点
    node.data=value
    node.next=None
    #检查是否为空队列
    if rear==None:
        front=node  #新建立的节点成为第1个节点
    else:
        rear.next=node  #将节点加入到队列的末尾
    rear=node  #将队列的末尾指针指向新加入的节点

#方法dequeue:队列数据的取出
def dequeue(action):
    global front
    global rear
    #从队列前端取出数据
    if not(front==None) and action==1:
        if front==rear:
            rear=None
        value=front.data  #将队列数据从前端取出
        front=front.next  #将队列的前端指针指向下一个
        return value
    #从队列末尾取出数据
    elif not(rear==None) and action==2:
        startNode=front  #先记下队列前端的指针值
        value=rear.data  #取出队列当前末尾的数据
        #查找队列末尾节点的前一个节点
        tempNode=front
        while front.next!=rear and front.next!=None:
            front=front.next
            tempNode=front
        front=startNode  #记录从队列末尾取出数据后的队列前端指针
        rear=tempNode  #记录从队列末尾取出数据后的队列末尾指针
        #下一行程序是指当队列中仅剩下最后一个节点时,
        #取出数据后便将front和rear指向None
        if front.next==None or rear.next==None:
            front=None
            rear=None
        return value
    else:
        return -1
    
print('用链表来实现双向队列')
print('====================================')

ch='a'
while True:
    ch=input('加入请按 a,取出请按 d,结束请按 e:')
    if ch =='e':
        break
    elif ch=='a':
        item=int(input('加入的元素值:'))
        enqueue(item)
    elif ch=='d':
        temp=dequeue(1)
        print('从双向队列前端按序取出的元素数据值为：%d' %temp)
        temp=dequeue(2)
        print('从双向队列末尾按序取出的元素数据值为：%d' %temp)
    else:
        break