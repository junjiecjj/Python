class Node:
    def __init__(self):
        self.value=0
        self.left_Thread=0
        self.right_Thread=0
        self.left_Node=None
        self.right_Node=None

rootNode=Node()
rootNode=None

#将指定的值加入到线索二叉树 
def Add_Node_To_Tree(value):
    global rootNode
    newnode=Node()
    newnode.value=value
    newnode.left_Thread=0
    newnode.right_Thread=0
    newnode.left_Node=None
    newnode.right_Node=None
    previous=Node()
    previous.value=value
    previous.left_Thread=0
    previous.right_Thread=0
    previous.left_Node=None
    previous.right_Node=None
    #设置线索二叉树的开头节点
    if rootNode==None:
        rootNode=newnode
        rootNode.left_Node=rootNode
        rootNode.right_Node=None
        rootNode.left_Thread=0
        rootNode.right_Thread=1
        return
    #设置开头节点所指的节点
    current=rootNode.right_Node
    if current==None:
        rootNode.right_Node=newnode
        newnode.left_Node=rootNode
        newnode.right_Node=rootNode
        return
    parent=rootNode #父节点是开头节点
    pos=0 #设置二叉树中的行进方向
    while current!=None:
        if current.value>value:
            if pos!=-1:
                pos=-1
                previous=parent
            parent=current
            if current.left_Thread==1:
                current=current.left_Node
            else:
                current=None
        else:
            if pos!=1:
                pos=1
                previous=parent
            parent=current
            if current.right_Thread==1:
                current=current.right_Node
            else:
                current=None
    if parent.value>value:
        parent.left_Thread=1
        parent.left_Node=newnode
        newnode.left_Node=previous
        newnode.right_Node=parent
    else:
        parent.right_Thread=1
        parent.right_Node=newnode
        newnode.left_Node=parent
        newnode.right_Node=previous
    return
            
    
#线索二叉树中序遍历
def trace():
    global rootNode
    tempNode=rootNode
    while True: 
        if tempNode.right_Thread==0:
            tempNode=tempNode.right_Node
        else:
            tempNode=tempNode.right_Node
            while tempNode.left_Thread!=0:
                tempNode=tempNode.left_Node
        if tempNode!=rootNode:
            print('[%d]' %tempNode.value)
        if tempNode==rootNode:
            break
#主程序    
i=0
array_size=11
print('线索二叉树经建立后,以中序遍历有排序的效果')
print('第一个数字为线索二叉树的开头节点,不列入排序')
data1=[0,10,20,30,100,399,453,43,237,373,655]
for i in range(array_size):
    Add_Node_To_Tree(data1[i])
print('====================================')
print('范例 1 ')
print('数字从小到大的排序结果为: ')
trace()

data2=[0,101,118,87,12,765,65]
rootNode=None  #将线索二叉树的树根归零
array_size=7   #第2个范例的数组长度为7
for i in range(array_size):
    Add_Node_To_Tree(data2[i])
print('====================================')
print('范例 2 ')
print('数字从小到大的排序结果为: ')
trace()
print()