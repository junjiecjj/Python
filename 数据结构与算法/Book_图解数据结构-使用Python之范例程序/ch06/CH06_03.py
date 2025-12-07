class tree:
    def __init__(self):
        self.data=0
        self.left=None
        self.right=None
        
def inorder(ptr):      #中序遍历子程序
    if ptr!=None:
        inorder(ptr.left)
        print('[%2d] ' %ptr.data, end='')
        inorder(ptr.right)

def create_tree(root,val):    #建立二叉树的函数
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

#主程序
data=[5,6,24,8,12,3,17,1,9]
ptr=None
root=None
for i in range(9):
    ptr=create_tree(ptr,data[i])       #建立二叉树
print('====================')
print('排序完成的结果：')
inorder(ptr)   #中序遍历
print('')