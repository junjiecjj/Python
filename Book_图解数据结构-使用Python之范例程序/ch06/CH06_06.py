class tree:
    def __init__(self):
        self.data=0
        self.left=None
        self.right=None

def create_tree(root,val):  #建立二叉树的函数
    newnode=tree()
    newnode.data=val
    newnode.left=None
    newnode.right=None
    if root==None:
        root=newnode
        return root
    else:
        current=root
        while current!=None:
            backup=current
            if current.data > val:
                current=current.left
            else:
                current=current.right
        if backup.data >val:
            backup.left=newnode
        else:
            backup.right=newnode
    return root

def search(ptr,val):     #在二叉树中查找某个值的子程序
    while True:
        if ptr==None:    #没找到就返回None
            return None
        if ptr.data==val:       #节点值等于查找值
            return ptr
        elif ptr.data > val:    #节点值大于查找值
            ptr=ptr.left
        else:
            ptr=ptr.right

def inorder(ptr):      #中序遍历子程序
    if ptr!=None:
        inorder(ptr.left)
        print('[%2d] ' %ptr.data, end='')
        inorder(ptr.right)

#主程序
arr=[7,1,4,2,8,13,12,11,15,9,5]
ptr=None
print('[原始数组内容]')
for i in range(11):
    ptr=create_tree(ptr,arr[i])  #建立二叉树
    print('[%2d] ' %arr[i],end='')
print()
data=int(input('请输入要查找的键值：'))
if search(ptr,data)!=None:      #在二叉树中查找
    print('二叉树中有此节点了!')
else:
    ptr=create_tree(ptr,data)
    inorder(ptr)