class list_node:
    def __init__(self):
        self.val=0
        self.next=None
        
head=[list_node()]*6 #声明一个节点类型的链表

newnode=list_node()


#图的数组声明
data=[[1,2],[2,1],[2,5],[5,2], \
      [2,3],[3,2],[2,4],[4,2], \
      [3,4],[4,3],[3,5],[5,3], \
      [4,5],[5,4]]

print('图的邻接表内容：')
print('----------------------------------')
for i in range(1,6):
    head[i].val=i  #链表头head
    head[i].next=None
    print('顶点 %d =>' %i,end='')  #把顶点值打印出来
    ptr=head[i]
    for j in range(14):    #遍历图的数组
        if data[j][0]==i:  #如果节点值=i，加入节点到链表头
            newnode.val=data[j][1]  #声明新节点，值为终点值
            newnode.next=None
            while ptr!=None:  #判断是否为链表的末尾
                ptr=ptr.next
            ptr=newnode       #加入新节点
            print('[%d] ' %newnode.val,end='')  #打印相邻顶点
    print()