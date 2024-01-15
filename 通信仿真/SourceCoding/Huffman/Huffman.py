#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 21:53:12 2023

@author: jack



(1) 给定N个权值作为N个叶子结点，构造一棵二叉树，若该树的带权路径长度达到最小，称这样的二叉树为最优二叉树，也称为哈夫曼树(Huffman Tree)。哈夫曼树是带权路径长度最短的树，权值较大的结点离根较近。
(2) 1952年，哈夫曼提出了一种构造最佳码的方法称为哈夫曼码（也有说霍夫曼码，看你怎么翻译了）。它充分利用了信源概率分布的特性进行编码，是无失真信源编码方法的一种。
(3) 为了设计长短不等的编码，以便减少电文的总长，还必须考虑编码的唯一性，即在建立不等长编码时必须使任何一个字符的编码都不是另一个字符的前缀，这宗编码称为前缀编码（prefix code）。


构建哈夫曼树
假设有n个权值，则构造出的哈夫曼树有n个叶子结点。 n个权值分别设为 w1、w2、…、wn，则哈夫曼树的构造规则为：
    (1) 将w1、w2、…，wn看成是有n 棵树的森林(每棵树仅有一个结点)；
    (2) 在森林中选出两个根结点的权值最小的树合并，作为一棵新树的左、右子树，且新树的根结点权值为其左、右子树根结点权值之和；
    (3) 从森林中删除选取的两棵树，并将新树加入森林；
    (4) 重复(2)、(3)步，直到森林中只剩一棵树为止，该树即为所求得的哈夫曼树。

哈夫曼编码
等长编码：这种编码方式的特点是每个字符的编码长度相同（编码长度就是每个编码所含的二进制位数）。假设字符集只含有4个字符A，B，C，D，用二进制两位表示的编码分别为00，01，10，11。若现在有一段电文为：ABACCDA，则应发送二进制序列：00010010101100，总长度为14位。当接收方接收到这段电文后，将按两位一段进行译码。这种编码的特点是译码简单且具有唯一性，但编码长度并不是最短的。

不等长编码：在传送电文时，为了使其二进制位数尽可能地少，可以将每个字符的编码设计为不等长的，使用频度较高的字符分配一个相对比较短的编码，使用频度较低的字符分配一个比较长的编码。例如，可以为A，B，C，D四个字符分别分配0，00，1，01，并可将上述电文用二进制序列：000011010发送，其长度只有9个二进制位，但随之带来了一个问题，接收方接到这段电文后无法进行译码，因为无法断定前面4个0是4个A，1个B、2个A，还是2个B，即译码不唯一，因此这种编码方法不可使用。
因此，为了设计长短不等的编码，以便减少电文的总长，还必须考虑编码的唯一性，即在建立不等长编码时必须使任何一个字符的编码都不是另一个字符的前缀，这宗编码称为前缀编码（prefix code）。
    1) 利用字符集中每个字符的使用频率作为权值构造一个哈夫曼树；
    2) 从根结点开始，为到每个叶子结点路径上的左分支赋予0，右分支赋予1，并从根到叶子方向形成该叶子结点的编码.
————————————————------------------------------------------------

https://blog.csdn.net/lzq20115395/article/details/78906863

https://zhuanlan.zhihu.com/p/108845114

https://blog.csdn.net/amnesia_h/article/details/123671999

https://zhuanlan.zhihu.com/p/103908133

https://www.cnblogs.com/kentle/p/14725589.html



"""
import numpy  as  np
import os, sys
import math


# 统计字符出现频率，生成映射表
def count_frequency(text):
    chars = []
    ret = []

    for char in text:
        if char in chars:
            continue
        else:
            chars.append(char)
            ret.append((char, text.count(char)))
    return ret

#节点类
class Node(object):
    def __init__(self, name = None, value = None):
        self._name = name
        self._value = value
        self._left = None
        self._right = None
        return

## 哈夫曼树
class HuffmanTree(object):
    #根据Huffman树的思想：以叶子节点为基础，反向建立Huffman树
    def __init__(self, char_weights):
        self.Init = [Node(part[0],part[1]) for part in char_weights]  #根据输入的字符及其频数生成叶子节点
        while len(self.Init) != 1:
            self.Init.sort(key = lambda node:node._value, reverse = True)
            c = Node(value = (self.Init[-1]._value + self.Init[-2]._value))
            c._left = self.Init.pop(-1)
            c._right = self.Init.pop(-1)
            self.Init.append(c)
        self.root = self.Init[0]
        self.deep = np.arange(2 * int( math.ceil(math.log(len(char_weights), 2))) )   #self.deep用于保存每个叶子节点的Haffuman编码,range的值只需要不小于树的深度就行
        self.coodbook = {}
        return

    #用递归的思想生成码本
    def prefix(self, tree, length):
        node = tree
        if (not node):
            return
        elif node._name:
            # print(node._name + '的编码为:')
            tmp = ""
            for i in range(length):
                # print(self.deep[i])
                tmp += str(self.deep[i])
            self.coodbook[node._name] = tmp
            # print('\n')
            return
        self.deep[length] = 0
        self.prefix(node._left, length + 1)
        self.deep[length] = 1
        self.prefix(node._right, length + 1)
        return

    #生成哈夫曼编码
    def get_code(self):
        self.prefix(self.root, 0)
        return

# if __name__=='__main__':
#输入的是字符及其频数
char_weights = [('a',22), ('i',32), ('b',4), ('c',10), ('d',8), ('f',15), ('g',2)]
te = HuffmanTree(char_weights)
te.get_code()
for key, val in te.coodbook.items():
    print(f"{key} : {val}")





























































































































































































































































































































































































































































