#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np

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


# 节点类
class Node(object):
    def __init__(self, name = None, value = None):
        self.name = name
        self.value = value
        self.left = None
        self.right = None
        self.father = None
        return
    def is_left(self):
        return self.father.left == self


class HuffmanTree(object):
    def __init__(self, char_weights):
        ## 根据输入的字符及其频数生成叶子节点
        self.nodes = [Node(part[0], part[1]) for part in char_weights]
        #=============================================================
        ## 根据Huffman树的思想：以叶子节点为基础，反向建立Huffman树
        #=============================================================
        self.queue = self.nodes[:]  ## 通过使用 [ : ] 切片，可以浅拷贝整个列表，同样的，只对第一层实现深拷贝。在这里，对queue的任何元素的left/right/value/name/father操作都会反应到nodes,但是queue.pop()不会影响nodes.
        while len(self.queue) > 1:
            self.queue.sort(key = lambda node:node.value, reverse = True)
            left_node = self.queue.pop(-1)
            right_node = self.queue.pop(-1)
            father_node = Node(value = (left_node.value + right_node.value))
            father_node.left = left_node
            father_node.right = right_node
            left_node.father = father_node
            right_node.father = father_node
            self.queue.append(father_node)
        #========================
        self.queue[0].father = None
        self.root = self.queue[0]  # root就是Huffman码树的根节点
        #=========================================
        ## 产生码本
        self.coodbook = {}
        self.prefix_coodbook()
        return

    # 生成Huffman码本, 方式1, 无前缀码, 从叶子节点到根节点
    def prefix_coodbook(self, ):
        self.coodbook = {}
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            key = node.name
            self.coodbook[key] = ""
            while node != self.root:
                if node.is_left():
                    self.coodbook[key] = '0' + self.coodbook[key]
                else:
                    self.coodbook[key] = '1' + self.coodbook[key]
                node = node.father
        return

    # 生成Huffman码本, 方式2, 无前缀码, 递归生成
    def get_codebook(self):
        self.coodbook = {}
        self.deep = np.arange(2 * int( math.ceil(math.log(len(self.nodes), 2))) )   #self.deep用于保存每个叶子节点的Haffuman编码,range的值只需要不小于树的深度就行
        self.prefix_coodbook_recursion(self.root, 0)
        return
    def prefix_coodbook_recursion(self, tree, length):
        node = tree
        if (not node):
            return
        elif node.name:
            tmp = ""
            for i in range(length):
                tmp += str(self.deep[i])
            self.coodbook[node.name] = tmp
            return
        self.deep[length] = 0
        self.prefix_coodbook_recursion(node.left, length + 1)
        self.deep[length] = 1
        self.prefix_coodbook_recursion(node.right, length + 1)
        return

    ## 编码器
    def encoder(self, text,):
        ret = ''
        for char in text:
            ret += self.coodbook[char]
        return ret

    ## 译码器
    def decoder(self, bin_string):
        ret = ''
        node = self.root
        for binary in bin_string:
            if binary == '0':
                node = node.left
            else:
                node = node.right
            if node.name:
                ret += node.name
                node = self.root
        return ret

###=======================================================================



# if __name__ == '__main__':
text = str("hello, jack, oh, sorry, that terrible~~, USTC, I LOVE YOU!!!")
text = str("aaaaabbbbccccccccccddddddddfffffffffffffffgg")
# text = str("h")  ## 此代码无法对一个字母的信源编解码
char_frequency = count_frequency(text)

# char_frequency = [('a',22), ('i',32), ('b',4), ('c',10), ('d',8), ('f',15), ('g',2)]
hf = HuffmanTree(char_frequency)
hf.prefix_coodbook()
huffman_str = hf.encoder(text)
origin_str = hf.decoder(huffman_str, )
# print(f"{text}")
print(f"{huffman_str}")
print(f"{origin_str}")

print(f"Check {text == origin_str} !!!")
print(f"码率：{len(huffman_str)/len(text)}")






