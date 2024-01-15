# =============== Program Description ===============                             
# 程序目的： 用链表实现二叉运算树
# ===================================================
#节点类的声明

class TreeNode:  #二叉树的节点声明
    def __init__(self,value):
        self.value=value #节点数据
        self.left_Node=None #指向左子树
        self.right_Node=None #指向左右子树

#二叉查找树类声明
class Binary_Search_Tree:
    #建立空的二叉查找树
    def __init__(self):
        self.rootNode=None

    #传入一个数组来建立二叉树
    def __init__(self,data):
        for i in range(len(data)):
            self.Add_Node_To_Tree(data[i])

    #将指定的值加入到二叉树中适当的节点
    def Add_Node_To_Tree(self,value):
        currentNode=self.rootNode
        if self.rootNode==None:  #建立树根
            self.rootNode=TreeNode(value)
            return
        #建立二叉树
        while True:
            #符合这个判断表示此节点在左子树
            if value<currentNode.value:
                if currentNode.left_Node==None:
                    currentNode.left_Node=TreeNode(value)
                    return
                else:
                    currentNode=currentNode.left_Node
            #符合这个判断表示此节点在右子树
            else:
                if currentNode.right_Node==None:
                    currentNode.right_Node=TreeNode(value)
                    return
                else:
                    currentNode=currentNode.right_Node
                       
class Expression_Tree (Binary_Search_Tree):
    #初始化
    def __init__(self,information,index):
        #create方法可以将二叉树的数组表示法转换成链表表示法
        self.rootNode=self.create(information, index)

    # create方法的程序内容
    def create(self,sequence,index):
        if index >= len(sequence):   # 作为递归调用的出口条件
            return None
        else:
            tempNode = TreeNode((ord)(sequence[index]))
            # 建立左子树
            tempNode.left_Node = self.create(sequence, 2*index)
            # 建立右子树
            tempNode.right_Node =self.create(sequence, 2*index+1)
            return tempNode
        
    # preOrder(前序走访)方法的程序内容
    def preOrder(self,node):
        if node != None:
            print((chr)(node.value),end='')
            self.preOrder(node.left_Node)
            self.preOrder(node.right_Node)
            
    # inOrder(中序走访)方法的程序内容
    def inOrder(self,node):
        if node != None:
            self.inOrder(node.left_Node)
            print((chr)(node.value),end='')
            self.inOrder(node.right_Node)

    # postOrder(后序走访)方法的程序内容
    def postOrder(self,node):
        if node != None:
            self.postOrder(node.left_Node)
            self.postOrder(node.right_Node)
            print((chr)(node.value),end='')
        
    # 判断表达式如何运算的方法声明内容
    def condition(self,oprator, num1, num2):
        if oprator=='*' :
            return ( num1 * num2 ) #乘法请返回num1 * num2
        elif oprator=='/' :
            return ( num1 / num2 ) #除法请返回num1 / num2
        elif oprator=='+' :
            return ( num1 + num2 ) #加法请返回num1 + num2
        elif oprator=='-' :
            return ( num1 - num2 ) #减法请返回num1 - num2
        elif oprator=='%' :
            return ( num1 % num2 ) #取余数法请返回num1 % num2
        else:
            return -1
        
    #传入根节点,用来计算此二叉运算树的值
    def answer(self,node):
        firstnumber = 0
        secondnumber = 0
        #递归调用的出口条件
        if node.left_Node == None and node.right_Node == None :
            #将节点的值转换成数值后返回
            return node.value-48
        else:
            firstnumber = self.answer(node.left_Node)#计算左子树表达式的值
            secondnumber = self.answer(node.right_Node) #计算右子树表达式的值
            return self.condition((chr)(node.value), firstnumber, secondnumber)

#主程序

# 第一个表达式
information1 = [' ','+','*','%','6','3','9','5' ]

# 第二个表达式 
information2 = [' ','+','+','+','*','%','/','*',  \
                '1','2','3','2','6','3','2','2' ]

exp1 = Expression_Tree(information1, 1)
print('====二叉运算树数值运算范例 1: ====')
print('=================================')
print('===转换成中序法表达式===:  ',end='')
exp1.inOrder(exp1.rootNode)     
print('\n===转换成前序法表达式===:  ',end='')
exp1.preOrder(exp1.rootNode)    
print('\n===转换成后序法表达式===:  ',end='')
exp1.postOrder(exp1.rootNode)   

# 计算二叉树表达式的运算结果
print('\n此二叉运算树,经过计算后所得到的结果值: ',end='')
print(exp1.answer(exp1.rootNode))


# 建立第二棵二叉查找树对象
exp2 = Expression_Tree(information2, 1)
print()
print('====二叉运算树数值运算范例 2: ====')
print('=================================')
print('===转换成中序法表达式===:  ',end='')
exp2.inOrder(exp2.rootNode)     
print('\n===转换成前序法表达式===:  ',end='')
exp2.preOrder(exp2.rootNode)    
print('\n===转换成后序法表达式===:  ',end='')
exp2.postOrder(exp2.rootNode)   

# 计算二叉树表达式的运算结果
print('\n此二叉运算树,经过计算后所得到的结果值: ',end='')
print(exp2.answer(exp2.rootNode))