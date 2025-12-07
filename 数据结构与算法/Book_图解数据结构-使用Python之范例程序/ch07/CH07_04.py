class list_node:
    def __init__(self):
        self.val=0
        self.next=None

head=[list_node()]*9 #声明一个节点类型的链表数组
        
run=[0]*9

def dfs(current): #深度优先函数
    run[current]=1
    print('[%d] ' %current, end='')
    ptr=head[current].next
    while ptr!=None:
        if run[ptr.val]==0:        #如果顶点尚未遍历，
            dfs(ptr.val)           #就进行dfs的递归调用
        ptr=ptr.next
        
#声明图的边线数组       
data=[[1,2],[2,1],[1,3],[3,1], \
      [2,4],[4,2],[2,5],[5,2], \
      [3,6],[6,3],[3,7],[7,3], \
      [4,8],[8,4],[5,8],[8,5], \
      [6,8],[8,6],[8,7],[7,8]]
for i in range(1,9):  #共有八个顶点
    run[i]=0          #把所有顶点设置成尚未遍历过
    head[i]=list_node()
    head[i].val=i     #设置各个链表头的初值
    head[i].next=None
    ptr=head[i]        #设置指针指向链表头
    for j in range(20): #二十条边线
        if data[j][0]==i: #如果起点和链表头相等，则把顶点加入链表
            newnode=list_node()
            newnode.val=data[j][1]
            newnode.next=None
            while True:
                ptr.next=newnode    #加入新节点
                ptr=ptr.next
                if ptr.next==None:
                    break
        

print('图的邻接表内容：')      #打印图的邻接表内容
for i in range(1,9):
    ptr=head[i]
    print('顶点 %d=> ' %i,end='')
    ptr =ptr.next
    while ptr!=None:
        print('[%d] ' %ptr.val,end='')
        ptr=ptr.next
    print()
print('深度优先遍历的顶点：')      #打印深度优先遍历的顶点
dfs(1)
print()