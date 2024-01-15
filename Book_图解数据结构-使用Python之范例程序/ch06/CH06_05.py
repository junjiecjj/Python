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

def search(ptr,val):     #查找二叉树中某个值的子程序
    i=1
    while True:
        if ptr==None:    #没找到就返回None
            return None
        if ptr.data==val:       #节点值等于查找值
            print('共查找 %3d 次' %i)
            return ptr
        elif ptr.data > val:  #节点值大于查找值
            ptr=ptr.left
        else:
            ptr=ptr.right
        i+=1

#主程序
arr=[7,1,4,2,8,13,12,11,15,9,5]
ptr=None
print('[原始数组内容]')
for i in range(11):
    ptr=create_tree(ptr,arr[i])  #建立二叉树
    print('[%2d] ' %arr[i],end='')
print()
data=int(input('请输入查找值：'))
if search(ptr,data) !=None :    #在二叉树中查找
    print('您要找的值 [%3d] 找到了!!' %data)
else:
    print('您要找的值没找到!!')