#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:57:08 2023

@author: jack
"""
##==============================================================================================
## 使用自带的list实现栈
##==============================================================================================

stack = [3, 4, 5]
# 入栈
stack.append(6)
stack.append(7)
# stack
# [3, 4, 5, 6, 7]

# 出栈
stack.pop()
# 7
# stack
# [3, 4, 5, 6]
stack.pop()
# 6



##==============================================================================================
## 使用 list 实现栈
##==============================================================================================

class Stack(object):
    """栈，使用顺序表实现"""

    def __init__(self):
        """定义空栈"""
        self.__items = []

    def is_empty(self):
        """判断栈是否为空"""
        return self.__items == []

    def push(self, item):
        """添加一个新元素item到栈顶"""
        self.__items.append(item)

    def pop(self):
        """弹出栈顶元素，有返回值"""
        return self.__items.pop()

    def peek(self):
        """返回栈顶元素，有返回值"""
        # 判断栈是否为空
        if self.is_empty():
            return None
        else:
            return self.__items[-1]

    def size(self):
        """返回栈中的元素个数"""
        return len(self.__items)

    def travel(self):
        """遍历栈中的元素"""
        for item in self.__items:
            print(item)


if __name__ == '__main__':
    s = Stack()
    s.push(7)
    s.push(8)
    s.push(9)
    s.pop()
    print(s.size())
    print(s.is_empty())
    print(s.peek())
    s.travel()















