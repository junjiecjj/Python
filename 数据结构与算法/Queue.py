#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:57:08 2023

@author: jack

"""



##==============================================================================================
## 使用自带的 Queue
##==============================================================================================


"""
queue
Python的Queue模块提供一种适用于多线程编程的先进先出(FIFO)容器
使用:put(),将元素添加到序列尾端，get(),从队列尾部移除元素。
 Queue.Queue(maxsize=0)   FIFO， 如果maxsize小于1就表示队列长度无限
       Queue.LifoQueue(maxsize=0)   LIFO， 如果maxsize小于1就表示队列长度无限
       Queue.qsize()   返回队列的大小
       Queue.empty()   如果队列为空，返回True,反之False
       Queue.full()   如果队列满了，返回True,反之False
       Queue.get([block[, timeout]])   读队列，timeout等待时间
       Queue.put(item, [block[, timeout]])   写队列，timeout等待时间
       Queue.queue.clear()   清空队列

"""

from queue import Queue

q = Queue()

for i in range(3):
    q.put(i)

while not q.empty():
    print(q.get())

#
# 0
# 1
# 2




# LifoQueue
# 使用后进先出序
from queue import LifoQueue

q = LifoQueue()

for i in range(3):
    q.put(i)

while not q.empty():
    print(q.get())
#
# 2
# 1
# 0


# 优先队列：PriorityQueue,
# 依据队列中内容的排序顺序(sort order)来决定那个元素将被检索
from queue import PriorityQueue


class Job(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
        print('New job:', description)
        return

    def __lt__(self, other):
        #定义小于操作符(<)的行为
        return self.priority < other.priority

q = PriorityQueue()

#q.put((数字,值)),特点：数字越小，优先级越高
q.put(Job(5, 'Mid-level job'))
q.put(Job(10, 'Low-level job'))
q.put(Job(1, 'Important job'))

while not q.empty():
    next_job = q.get()
    print('Processing job', next_job.description)

#
# New job: Mid-level job
# New job: Low-level job
# New job: Important job
# Processing job: Important job
# Processing job: Mid-level job
# Processing job: Low-level job



##==============================================================================================
## 使用自带的 list 实现 队列
##==============================================================================================

class Queue(object):
    def __init__(self):
        """初始化，即定义一个空队列"""
        self.__items = []

    def is_empty(self):
        """判空操作，判断队列是否为空"""
        return self.__items == []

    def enqueue(self, item):
        """向队列中添加新元素"""
        self.__items.append(item)

    def dequeue(self):
        """返回队列中的第一个元素"""
        return self.__items.pop(0)

    def size(self):
        """返回队列中的元素个数"""
        return len(self.__items)

    def travel(self):
        """遍历队列中的元素"""
        for item in self.__items:
            print(item)


if __name__ == '__main__':
    q = Queue()
    print(q.is_empty())
    q.enqueue(7)
    q.enqueue(8)
    q.enqueue(9)
    print(q.is_empty())
    print(q.size())
    print(q.dequeue())
    q.dequeue()
    q.travel()



##==============================================================================================
## 使用自带的 list 实现 双端队列
##==============================================================================================


class Dequeue(object):
    """初始化队列"""

    def __init__(self):
        self.__items = []

    def is_empty(self):
        """判断队列是否为空"""
        return self.__items == []

    def size(self):
        """返回队列中元素个数"""
        return len(self.__items)

    def add_front(self, item):
        """从队头添加一个新元素item"""
        self.__items.insert(0, item)

    def add_rear(self, item):
        """从队尾添加一个新元素item"""
        self.__items.append(item)

    def remove_front(self):
        """从队头删除一个元素"""
        return self.__items.pop(0)

    def remove_rear(self):
        """从队尾删除一个元素"""
        return self.__items.pop()

    def travel(self):
        """从队头遍历队列"""
        for item in self.__items:
            print(item)


if __name__ == '__main__':
    d = Dequeue()
    print(d.is_empty())
    d.add_front(7)
    d.add_front(8)
    d.add_rear(9)
    d.travel()
    d.remove_front()
    d.travel()
    print(d.size())


##==============================================================================================
## 使用自带的 deque 实现队列和栈
##==============================================================================================


"""
Python collections模块之 deque 详解
标准库的 collections 模块中的 deque 结构体，它被设计成在两端存入和读取都很快的特殊 list，可以用来实现栈和队列的功能。
deque()
deque是栈和队列的一种广义实现，deque是"double-end queue"的简称；deque支持线程安全、有效内存地以近似O(1)的性能在deque的两端插入和删除元素，尽管list也支持相似的操作，但是它主要在固定长度操作上的优化，从而在pop(0)和insert(0,v)（会改变数据的位置和大小）上有O(n)的时间复杂度。


deque()
    append()
    appendleft()
    extend()
    extendleft()
    pop()
    popleft()
    count()
    insert(index,obj)
    rotate(n)
    clear()
    remove()
    maxlen
常用方法：
    append(): 从右端添加元素（与list同）
    appendleft():从左端添加元素
    extend():从右端逐个添加可迭代对象（与list同）Python中的可迭代对象有：列表、元组、字典、字符串
    extendleft():从左端逐个添加可迭代对象. Python中的可迭代对象有：列表、元组、字典、字符串
    pop():移除列表中的一个元素（默认最右端的一个元素），并且返回该元素的值（与list同），如果没有元素，将会报出IndexError.
    popleft():移除列表中的一个元素（默认最左端的一个元素），并且返回该元素的值，如果没有元素，将会报出IndexError.
    count():统计队列中的元素个数（与list同）.
    insert(index,obj):在指定位置插入元素（与list同）
    rotate(n):rotate(n)， 从右侧反转n步，如果n为负数，则从左侧反转。d.rotate(1) 等于 d.appendleft(d.pop())
    clear():将deque中的元素全部删除，最后长度为0
    remove():移除第一次出现的元素，如果没有找到，报出ValueError
    maxlen:只读的属性，deque限定的最大长度，如果无，就返回None。当限制长度的deque增加超过限制数的项时, 另一边的项会自动删除。
    此外,deque还支持迭代、序列化、len(d)、reversed(d)、copy.copy(d)、copy.deepcopy(d)，通过in操作符进行成员测试和下标索引。


"""
from collections import deque

##===============================================
## append(): 从右端添加元素（与list同）
##===============================================

st = "abcd"
list1 = [0, 1, 2, 3]
dst = deque(st)
dst.append(4)
dst.append([22,33])

dlist1 = deque(list1)
dlist1.append("k")

print(dst)
print(dlist1)
#结果：
#deque(['a', 'b', 'c', 'd', 4])
#deque([0, 1, 2, 3, 'k'])

import copy
print(f"len(dlist1) = {len(dlist1)}")
print(f"reversed(dlist1) = {reversed(dlist1)}")
print(f"copy.deepcopy(d) = {copy.deepcopy(dlist1)}")
##===============================================
## appendleft():从左端添加元素
##===============================================
st = "abcd"
list1 = [0, 1, 2, 3]
dst = deque(st)
dst.appendleft(4)

dlist1 = deque(list1)
dlist1.appendleft("k")
print(dst)
print(dlist1)
#结果：
#deque([4, 'a', 'b', 'c', 'd'])
#deque(['k', 0, 1, 2, 3])



##===============================================
## extend():从右端逐个添加可迭代对象（与list同）Python中的可迭代对象有：列表、元组、字典、字符串
##===============================================
from collections import deque
ex = (1, "h", 3)
st = "abcd"
dst = deque(st)
dst.extend(ex)

list1 = [0, 1, 2, 3]
dlist1 = deque(list1)
dlist1.extend(ex)
print(dst)
print(dlist1)
#结果：
#deque(['a', 'b', 'c', 'd', 1, 'h', 3])
#deque([0, 1, 2, 3, 1, 'h', 3])


##===============================================
## extendleft():从左端逐个添加可迭代对象. Python中的可迭代对象有：列表、元组、字典、字符串
##===============================================

from collections import deque

ex = [("a", 1), 3]
st = "abcd"
dst = deque(st)
dst.extend(ex)

list1 = [0, 1, 2, 3]
dlist1 = deque(list1)
dlist1.extend(ex)
print(dst)
print(dlist1)
#结果：
#deque(['a', 'b', 'c', 'd', ('a', 1), 3])
#deque([0, 1, 2, 3, ('a', 1), 3])

##===============================================
## pop():移除列表中的一个元素（默认最右端的一个元素），并且返回该元素的值（与list同），如果没有元素，将会报出IndexError.
##===============================================


from collections import deque
st = "abcd"
dst = deque(st)
print(dst)
p = dst.pop()

list1 = [0, 1, 2, 3]
dlist1 = deque(list1)
print(dlist1)
p1 = dlist1.pop()


print(p)
print(p1)
print(dst)
print(dlist1)
#结果:
#d
#3
#deque(['a', 'b', 'c'])
#deque([0, 1, 2])

##===============================================
## popleft():移除列表中的一个元素（默认最左端的一个元素），并且返回该元素的值，如果没有元素，将会报出IndexError.
##===============================================


from collections import deque
st = "abcd"
dst = deque(st)
p = dst.popleft()


list1 = [0, 1, 2, 3]
dlist1 = deque(list1)
p1 = dlist1.popleft()
print(p)
print(p1)
print(dst)
print(dlist1)
#结果:
#a
#0
#deque(['b', 'c', 'd'])
#deque([1, 2, 3])

##===============================================
## count():统计队列中的元素个数（与list同）.
##===============================================


from collections import deque

st = "abbcd"
dst = deque(st)
p = dst.count("b")
print(dst)
print(p)
#结果:
#deque(['a', 'b', 'b', 'c', 'd'])
#2

##===============================================
## insert(index,obj):在指定位置插入元素（与list同）
##===============================================

from collections import deque

st = "abbcd"
dst = deque(st)
dst.insert(0, "chl")
print(dst)
#结果:
#deque(['chl', 'a', 'b', 'b', 'c', 'd'])

##===============================================
## rotate(n):rotate(n)， 从右侧反转n步，如果n为负数，则从左侧反转。d.rotate(1) 等于 d.appendleft(d.pop())
##===============================================


from collections import deque

st = "abbcd"
dst = deque(st)
dst.rotate(1)
print(dst)
#结果:
#deque(['d', 'a', 'b', 'b', 'c'])


##===============================================
##   clear():将deque中的元素全部删除，最后长度为0
##===============================================

from collections import deque

st = "abbcd"
dst = deque(st)
dst.clear()
print(dst)
#结果:
#deque([])

##===============================================
## remove():移除第一次出现的元素，如果没有找到，报出ValueError
##===============================================

from collections import deque

st = "abbcd"
dst = deque(st)
dst.remove("a")
print(dst)
dst.remove("f")
#结果:
#deque(['b', 'b', 'c', 'd'])
#ValueError: deque.remove(x): x not in deque

##===============================================
## maxlen:只读的属性，deque限定的最大长度，如果无，就返回None。
##===============================================

from collections import deque

dst = deque(maxlen=2)
dst.append(1)
dst.append(2)
print(dst)
dst.append(3)
print(dst)
print(dst.maxlen)
#结果:
#deque([1, 2], maxlen=2)
#deque([2, 3], maxlen=2)
#2







































