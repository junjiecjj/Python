

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:12:25 2023

@author: jack
(一 )相关定义：
    (1) 满二叉树是特殊的二叉树，它要求除叶结点外的其他结点都具有两棵子树，并且所有的叶结点都在同一层上。
    (2) 完全二叉树是特殊的二叉树，若完全二叉树具有个结点，它要求个结点与满二叉树的前个结点具有完全相同的逻辑结构。
    (3) 二叉搜索树：小于父节点的键都在左子树中，大于父节点的键则都在右子树中。
    (4) 二叉搜索树（Binary Search Tree）也叫二叉查找树，他是具有下列性质的一种二叉树。
        若左子树不空，则左子树上所有节点的值都小于根节点的值；
        若右子树不空，则右子树上所有节点的值都大于根节点的值；
        任意节点的子树也都是二叉搜索树；


(二) 相关性质
    对于完全二叉树，如果有N个节点，则此二叉树的深度为 [log2(N+1)]，此处的[]为向下取整。


(三) 二叉树的遍历
    二叉树的遍历是指沿着某条搜索路径访问二叉树的结点，每个结点被访问的次数有且仅有一次。
    二叉树通常可划分为三个部分，即根结点、左子树和右子树。根据三个部分的访问顺序不同，可将二叉树的遍历方法分为以下几种。
    (1) 前序遍历：先访问根结点，再前序遍历左子树，最后前序遍历右子树。
    (2) 中序遍历：先中序遍历左子树，再访问根结点，最后中序遍历右子树。
    (3) 后序遍历：先后序遍历左子树，再后序遍历右子树，最后访问根结点。
    (4) 层次遍历：自上而下、从左到右依次访问每层的结点。
    二叉树遍历操作的递归算法结构简洁，易于实现，但是在时间上开销较大，运行效率较低，为了解决这一问题，可以将递归算法转换为非递归算法，转换方式有以下两种：
    使用临时遍历保存中间结果，用循环结构代替递归过程。利用栈保存中间结果。二叉树遍历操作实现的非递归算法利用栈结构通过回溯访问二叉树的每个结点。

"""


from queue import Queue



class node(object):
    def __init__(self, val = None, name = None):
        self.value = val
        self.name = name
        self.left = None
        self.right = None
        self.father = None
        return
    def is_left(self):
        return self.father.left == self


# https://zhuanlan.zhihu.com/p/539690796
# 二叉树类, 准确说叫二叉搜索树，左节点<根节点<右节点
class BinarySearchTree(object):
    def __init__(self, ):
        self.root = None     # 二叉树的根结点
        self.size = 0
        self.pre = []
        self.In = []
        self.post = []
        return

    ## 插入（非递归）
    def insert(self, val):
        newnode = node(val = val)
        if not self.root:
            self.root = newnode
            self.size += 1
            return True
        else:
            tmp = self.root
            while tmp:
                if tmp.value > val:
                    if tmp.left != None:
                        tmp = tmp.left
                    else:
                        tmp.left = newnode
                        self.size += 1
                        return True
                else:
                    if tmp.right != None:
                        tmp = tmp.right
                    else:
                        tmp.right = newnode
                        self.size += 1
                        return True

    ## 查找节点（非递归）
    def search(self, val):
        tmp = self.root
        while 1:
            if tmp == None:
                return None
            if tmp.value == val:
                return tmp
            elif tmp.value > val:
                tmp = tmp.left
            else:
                tmp = tmp.right

    ## 查找节点（递归）
    def searchBST(self, root, val):
        if root == None:
            return None
        elif val == root.value:
            return root
        elif val < root.value:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

    ##====================================前序遍历========================================
    ## 前序遍历（递归）
    def preOrder(self, root):
        if  root is not None:
            print( root.value, end=' ')
            self.pre.append(root.value)
            self.preOrder(root.left)
            self.preOrder(root.right)
        return

    ## 前序遍历（非递归）
    def preorder_traversal(self, root):
        ret = []
        stack = []
        while root or stack:
            while root:
                ret.append(root.value)
                stack.append(root)
                root = root.left
            if stack:
                t = stack.pop()
                root = t.right
        return ret

    ## 前序遍历（非递归）
    def pre_order(self, root):
        if not root: # 空
            return []
        stack, result = [], []
        stack.append(root)
        while stack:
            node = stack.pop() # 先序 遇到就先输出
            result.append(node.value)
            # 栈的弹出顺序与入栈顺序相反 因此先入右再入左
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return result

    ##====================================中序遍历========================================
    ## 中序遍历（递归）
    def inOrder(self, root):
        if  root is not None:
            self.inOrder(root.left)
            print(root.value, end=' ')
            self.In.append(root.value)
            self.inOrder(root.right)
        return

    ## 中序遍历（非递归）
    def inorder_traversal(self, root):
        if not root:
            return
        stack = []
        result = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            if stack:
                t = stack.pop()
                result.append(t.value)
                root = t.right
        return result

    def mid_order(self, head):
        '''
        二叉树中序遍历非递归实现
        1.当前节点如有左节点就不断压栈 无左节点就可以出栈打印
        2.打印出栈的节点若有右子树则将右节点执行1的步骤
        :param head: 头结点
        :return: result列表类型
        '''
        if not head: #空
            return []
        stack, result = [], []
        node = head
        ## 循环条件：1.栈非空则还可以输出 2.栈空但是节点非空说明还有节点可以压栈
        while node or stack:
            if not node: # 如果节点为空 证明没有左子树 弹出一个
                node = stack.pop()
                result.append(node.value)
                node = node.right #尝试是否有右节点
            else: # 节点非空 压栈 尝试是否有左子树
                stack.append(node)
                node = node.left
        return result

    ##====================================后序遍历========================================
    ## 后序遍历（递归）
    def postOrder(self, root):
        if  root is not None:
            self.postOrder(root.left)
            self.postOrder(root.right)
            print(root.value, end=' ')
            self.post.append(root.value)
        return

    ## 后序遍历（非递归）
    def postOrder_traversal(self, root):
        ret = []
        stack = []
        while root or stack:
            while root:
                ret.append(root.value)
                stack.append(root)
                root = root.right
            if stack:
                t = stack.pop()
                root = t.left
        return  ret[::-1]

    ## 后序遍历（非递归）
    def post_order(self, head):
        '''
        后序遍历二叉树非递归后序遍历是 左-右-中
        反过来就是 中-右-左 其实就是先序遍历镜像二叉树(即左右互换)
        :param head: 头节点
        :return: result[::-1] 逆序
        '''
        if not head:
            return []
        stack, result = [], []
        stack.append(head)
        while len(stack) != 0:
            node = stack.pop() # 先序 遇到就先输出
            result.append(node.value)
            # 先压栈左节点再压右节点 所以输出就是先右后左
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return result[::-1] # 将 中-右-左 逆序变为 左-右-中
    ##====================================层次遍历========================================
    ## 层次遍历（非递归）
    def order(self, root):

        return
    ## 层次遍历（非递归）
    def level_traversal(self, root):
        if root is None:
            return
        queue = [root]
        end = queue[-1]
        level = 1
        print(f"level {level}:"  )
        while len(queue) != 0:
            cur = queue.pop(0)
            print(cur.value)
            if cur.left is not None:
                queue.append(cur.left)
            if cur.right is not None:
                queue.append(cur.right)
            if cur == end:
                level += 1
                if len(queue) != 0:
                    end = queue[-1]
                    print(f"level {level}:" )
        return

bt = BinarySearchTree()
bt.insert(10)
bt.insert(2)
bt.insert(1)
# bt.insert(1)
bt.insert(12)
bt.insert(16)

bt.insert(7)
bt.insert(11)

print("前序遍历\n")
bt.preOrder(bt.root)
pre = bt.preorder_traversal(bt.root)
print(f"\n{pre}\n ")
pre = bt.pre_order(bt.root)
print(f"{pre}\n ")

print("中序遍历\n")
bt.inOrder(bt.root)
In = bt.inorder_traversal(bt.root)
print(f"\n{In}\n ")
In = bt.mid_order(bt.root)
print(f"{In}\n")


print("后序遍历\n")
bt.postOrder(bt.root)
post = bt.postOrder_traversal(bt.root)
print(f"\n{post} \n ")
post = bt.post_order(bt.root)
print(f"{post} \n ")


print("层次遍历\n")
bt.level_traversal(bt.root)
bt.order(bt.root)
print("\n")































