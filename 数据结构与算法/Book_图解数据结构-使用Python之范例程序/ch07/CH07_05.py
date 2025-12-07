MAXSIZE=10  #定义队列的最大容量	

front=-1 #指向队列的前端
rear=-1  #指向队列的末尾

class Node:
    def __init__(self,x):
        self.x=x        #顶点数据
        self.next=None  #指向下一个顶点的指针
        
class GraphLink:
    def __init__(self):
        self.first=None
        self.last=None
        
    def my_print(self):
        current=self.first
        while current!=None:
            print('[%d]' %current.x,end='')
            current=current.next
        print()

    def insert(self,x):
        newNode=Node(x)
        if self.first==None:
            self.first=newNode
            self.last=newNode
        else:
            self.last.next=newNode
            self.last=newNode
 
#队列数据的存入
def enqueue(value):
    global MAXSIZE
    global rear
    global queue
    if rear>=MAXSIZE:
        return
    rear+=1
    queue[rear]=value
    

#队列数据的取出
def dequeue():
    global front
    global queue
    if front==rear:
        return -1
    front+=1
    return queue[front]

#广度优先查找法
def bfs(current):
    global front
    global rear
    global Head
    global run
    enqueue(current) #将第一个顶点存入队列
    run[current]=1   #将遍历过的顶点设置为1
    print('[%d]' %current, end='') #打印出该遍历过的顶点
    while front!=rear:             #判断当前的队伍是否为空
        current=dequeue()            #将顶点从队列中取出
        tempnode=Head[current].first #先记录当前顶点的位置
        while tempnode!=None:
            if run[tempnode.x]==0:
                enqueue(tempnode.x)
                run[tempnode.x]=1   #记录已遍历过
                print('[%d]' %tempnode.x,end='')
            tempnode=tempnode.next

#声明图的边线数组
Data=[[0]*2 for row in range(20)]

Data =[[1,2],[2,1],[1,3],[3,1],[2,4], \
       [4,2],[2,5],[5,2],[3,6],[6,3], \
       [3,7],[7,3],[4,5],[5,4],[6,7],[7,6],[5,8],[8,5],[6,8],[8,6]]

run=[0]*9 #用来记录各顶点是否遍历过
queue=[0]*MAXSIZE
Head=[GraphLink]*9
 			
print('图的邻接表内容：') #打印图的邻接表内容
for i in range(1,9):      #共有8个顶点
    run[i]=0              #把所有顶点设置成尚未遍历过
    print('顶点%d=>' %i,end='')
    Head[i]=GraphLink()
    for j in range(20):
        if Data[j][0]==i: #如果起点和链表头相等，则把顶点加入链表
            DataNum = Data[j][1]
            Head[i].insert(DataNum)
    Head[i].my_print()    #打印图的邻接标内容

print('广度优先遍历的顶点：') #打印广度优先遍历的顶点
bfs(1)
print()