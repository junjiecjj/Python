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
        
def postorder(ptr):  #后序遍历
    if ptr!=None:
        postorder(ptr.left)
        postorder(ptr.right)
        print('[%2d] ' %ptr.data, end='')

def preorder(ptr):   #前序遍历
    if ptr!=None:
        print('[%2d] ' %ptr.data, end='')
        preorder(ptr.left)
        preorder(ptr.right)

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
        
#主程序
data=[7,4,1,5,16,8,11,12,15,9,2]
ptr=None
root=None
for i in range(11):
    ptr=create_tree(ptr,data[i])       #建立二叉树
print('=======================================================')
print('中序遍历的结果：')
inorder(ptr)   #中序遍历
print()
print('=======================================================')
print('后序遍历的结果：')
postorder(ptr)   #后序遍历
print()
print('=======================================================')
print('前序遍历的结果：')
preorder(ptr)   #前序遍历
print()	