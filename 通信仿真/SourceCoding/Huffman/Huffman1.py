#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:34:32 2023

@author: jack
"""


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
class Node:
    def __init__(self, frequency):
        self.left = None
        self.right = None
        self.father = None
        self.frequency = frequency
    def is_left(self):
        return self.father.left == self
    def is_right(self):
        return self.father.right == self

# 创建叶子节点
def create_nodes(frequency_list):
    return [Node(frequency) for frequency in frequency_list]


# 创建Huffman树
def create_huffman_tree(nodes):
    ## 通过使用 [ : ] 切片，可以浅拷贝整个列表，同样的，只对第一层实现深拷贝。在这里，对queue的任何操作都会反应到nodes。
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key = lambda item: item.frequency)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.frequency + node_right.frequency)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]


# Huffman码本
def huffman_encoding(nodes, root):
    coodbook = [''] * len(nodes)
    for i in range(len(nodes)):
        node = nodes[i]
        while node != root:
            if node.is_left():
                coodbook[i] = '0' + coodbook[i]
            else:
                coodbook[i] = '1' + coodbook[i]
            node = node.father
    return coodbook


# 编码整个字符串
def encode_str(text, char_frequency, coodbook):
    ret = ''
    for char in text:
        i = 0
        for item in char_frequency:
            if char == item[0]:
                ret += coodbook[i]
            i += 1
    return ret


# 解码整个字符串
def decode_str(huffman_str, char_frequency, coodbook):
    ret = ''
    while huffman_str != '':
        i = 0
        for item in coodbook:
            if item in huffman_str and huffman_str.index(item) == 0:
                ret += char_frequency[i][0]
                huffman_str = huffman_str[len(item):]
            i += 1

    return ret


# if __name__ == '__main__':
text = str("hello, jack, i want fuck you, bitch, you are pussy ")

char_frequency = count_frequency(text)
nodes = create_nodes([item[1] for item in char_frequency])
root = create_huffman_tree(nodes)
coodbook = huffman_encoding(nodes, root)

huffman_str = encode_str(text, char_frequency, coodbook)
origin_str = decode_str(huffman_str, char_frequency, coodbook)

print('text:\n' + text)
print('Encode result:\n' + huffman_str)
print('Decode result:\n' + origin_str)







