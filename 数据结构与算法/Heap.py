#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:57:08 2023

@author: jack

基本概念：
1、完全二叉树：若二叉树的深度为h，则除第h层外，其他层的结点全部达到最大值，且第h层的所有结点都集中在左子树。
2、满二叉树：满二叉树是一种特殊的的完全二叉树，所有层的结点都是最大值。

什么是堆？
    堆（英语：heap)是计算机科学中一类特殊的数据结构的统称。堆通常是一个可以被看做一棵树的数组对象。堆总是满足下列性质：
    堆中某个节点的值总是不大于或不小于其父节点的值；
    [堆总是一棵完全二叉树]。这一点非常重要，是堆建立过程的根据。
    [堆的左右孩子没有大小的顺序].
    将根节点最大的堆叫做最大堆或大根堆，根节点最小的堆叫做最小堆或小根堆。常见的堆有二叉堆、斐波那契堆等。


堆是一种特殊的二叉树数据结构，具有以下特点：
    堆顶元素（通常是最小元素）可快速访问和删除。
    每个节点的值总是小于等于（最小堆）或大于等于（最大堆）其子节点的值。
    最小堆通常用于实现优先队列，而最大堆通常用于堆排序。

堆是一种常用的树形结构，是一种特殊的完全二叉树，当且仅当满足所有节点的值总是不大于或不小于其父节点的值的完全二叉树被称之为堆。堆的这一特性称之为堆序性。
Heap: 堆。一般也称作Priority Queue(即优先队列）
因此，在一个堆中，根节点是最大（或最小）节点。如果根节点最小，称之为小顶堆（或小根堆），如果根节点最大，称之为大顶堆（或大根堆）。

Python 的 heapq 包实现的仅仅是最小堆！
heapq 实现了适用于 Python 列表的最小堆排序算法。
heapq模块提供了一系列函数来操作堆数据结构，包括：
    创建堆：可以用 list 初始化为 [ ]，也可以用 heapify ( ) 将一个 list 转化为堆。
    heapify()：将一个列表转换为最小堆。
    heappush(heap, item)：将 item 加入 heap，保持堆的不变性；
    heappop(heap)：弹出并返回 heap 的最小元素，用 heap[0] 可以只访问最小元素而不弹出；
    heappushpop(heap, item)：将 item 放入堆中，然后弹出并返回 heap 的最小元素，该函数比先调用 heappush() 再调用 heappop() 效率更高；
    heapreplace(heap, item)：先 pop 最小元素，再压入 item，不是相当于先调用 heappop() 再调用heappush()，而是直接把item替换最小值，再更新；
    heapify(x)：将 list x 原地转换成堆（线性时间）；
    nlargest(n, iterable, key=None)：返回 iterable 中最大的 n 个元素组成的 list；
    nsmallest(n, iterable, key=None)：返回 iterable 中最小的 n 个元素组成的 list；
    当 n 较小时，nlargest 和 nsmallest 函数效率较高；
    当 n 较大时，sorted() 函数效率更高；
    当 n == 1 时，用 min() 和 max() 效率更高。

堆是一个树状的数据结构，其中的子节点与父节点属于排序关系。可以使用列表或数组来表示二进制堆，使得元素 N 的子元素位于 2 * N + 1 和 2 * N + 2 的位置（对于从零开始的索引）。这种布局使得可以在适当的位置重新排列堆，因此在添加或删除数据时无需重新分配内存。

max-heap 确保父级大于或等于其子级。min-heap 要求父项小于或等于其子级。Python 的heapq模块实现了一个 min-heap。






"""
# https://blog.csdn.net/chandelierds/article/details/91357784





##=====================================================================================================
##  # https://zhuanlan.zhihu.com/p/79641424
##=====================================================================================================



import math
from io import StringIO



def show_tree(tree, total_width=36, fill=' '):
    """Pretty-print a tree."""
    output = StringIO()
    last_row = -1
    for i, n in enumerate(tree):
        # print(f"{i}, {n}")
        row = int(math.floor(math.log(i + 1, 2)))
        if row != last_row:
            output.write('\n')
        columns = 2 ** row
        col_width = int(math.floor(total_width / columns))
        output.write(str(n).center(col_width, fill))
        last_row = row
    print(output.getvalue())
    print('-' * total_width)
    print()



# [创建堆]
# 创建堆有两种基本方法：heappush() 和 heapify()。
# 当使用heappush()时，当新元素添加时，堆得顺序被保持了。
import heapq

data = [12, 5, 11, 22, 9, 2, 21, 28, 15, 1]
heap = []
print('random :', data)
print()

for n in data:
    print('add {:>3}:'.format(n))
    heapq.heappush(heap, n)
    show_tree(heap)
# [插入操作]
# 数字x插入是将x加入到该二叉树的最后一个节点，依据最小堆的定义，自底向上，递归调整。





# 如果数据已经在内存中，则使用 heapify() 来更有效地重新排列列表中的元素。
import heapq
data = [12, 5, 11, 22, 9, 2, 21, 28, 15, 1]

print('random    :', data)
heapq.heapify(data)
print('heapified :')
show_tree(data)




# [删除操作]
# 访问堆的内容
# 正确创建堆后，使用heappop()删除具有最小值的元素。
# 对于删除操作，是将二叉树的最后一个节点替换到根节点，然后自顶向下，递归调整。如果调整时遇到左右子孩子都比父节点小，则选择最小的那个对换。

import heapq

data = [12, 5, 11, 22, 9, 2, 21, 28, 15, 1]
print('random    :', data)
heapq.heapify(data)
print('heapified :')
show_tree(data)


for i in range(2):
    smallest = heapq.heappop(data)
    print('pop    {:>3}:'.format(smallest))
    show_tree(data)



# 要删除现有元素，并在一次操作中用新值替换它们，使用heapreplace()。
# heapq.heapreplace() 不是相当于先调用 heappop() 再调用heappush()，而是直接把item替换最小值，再更新；
import heapq

data = [12, 5, 11, 22, 9, 2, 21, 28, 15, 1]
heapq.heapify(data)
print('start:')
show_tree(data)

for n in [0, 13]:
    smallest = heapq.heapreplace(data, n)
    print('replace {:>2} with {:>2}:'.format(smallest, n))
    show_tree(data)


# heapq.heappushpop()是heappush和haeppop的结合，同时完成两者的功能，不是先进行heappush()，再进行heappop()，而是
import heapq

data = [12, 5, 11, 22, 9, 2, 21, 28, 15, 1]
heapq.heapify(data)
print('start:')
show_tree(data)

for n in [3, 13]:
    smallest = heapq.heappushpop(data, n)
    print('push {:>2} pop {:>2}:'.format(n, smallest))
    show_tree(data)






# [堆的数据极值]
# heapq 还包括两个函数来检查 iterable 并找到它包含的最大或最小值的范围。
import heapq

data = [12, 5, 11, 22, 9, 2, 21, 28, 15, 1]
print('all       :', data)
print('3 largest :', heapq.nlargest(3, data))
print('from sort :', list(reversed(sorted(data)[-3:])))
print('3 smallest:', heapq.nsmallest(3, data))
print('from sort :', sorted(data)[:3])


# [有效地合并排序序列]
# 将几个排序的序列组合成一个新序列对于小数据集来说很容易。
# list(sorted(itertools.chain(*data)))
# 对于较大的数据集，将会占用大量内存。不是对整个组合序列进行排序，而是使用 merge() 一次生成一个新序列。

import heapq
import random


random.seed(2016)

data = []
for i in range(4):
    new_data = list(random.sample(range(1, 101), 5))
    new_data.sort()
    data.append(new_data)

for i, d in enumerate(data):
    print('{}: {}'.format(i, d))

print('\nMerged:')
for i in heapq.merge(*data):
    print(i, end=' ')
print()



##=====================================================================================================
##  https://zhuanlan.zhihu.com/p/106170247
##=====================================================================================================



# 最大或最小的K个元素
import heapq

nums = [14, 20, 5, 28, 1, 21, 16, 22, 17, 28]
heapq.nlargest(3, nums)
# [28, 28, 22]
heapq.nsmallest(3, nums)
[1, 5, 14]



# 自定义排序
# 回到之前的内容，如果我们想要heapq排序的是一个对象。那么heapq并不知道应该依据对象当中的哪个参数来作为排序的衡量标准，所以这个时候，需要我们自己定义一个获取关键字的函数，传递给heapq，这样才可以完成排序。
# 比如说，我们现在有一批电脑，我们希望heapq能够根据电脑的价格排序：
laptops = [
    {'name': 'ThinkPad', 'amount': 100, 'price': 91.1},
    {'name': 'Mac', 'amount': 50, 'price': 543.22},
    {'name': 'Surface', 'amount': 200, 'price': 21.09},
    {'name': 'Alienware', 'amount': 35, 'price': 31.75},
    {'name': 'Lenovo', 'amount': 45, 'price': 16.35},
    {'name': 'Huawei', 'amount': 75, 'price': 115.65}
]

cheap = heapq.nsmallest(3, laptops, key=lambda s: s['price'])
expensive = heapq.nlargest(3, laptops, key=lambda s: s['price'])







##=====================================================================================================
##  https://zhuanlan.zhihu.com/p/669887657
# https://zhuanlan.zhihu.com/p/72248920
##=====================================================================================================
# 创建最小堆
import heapq

# 创建一个列表
data = [5, 7, 1, 3, 9, 2]

# 转换为最小堆
heapq.heapify(data)
print("Min Heap:", data)
# 向堆中添加元素

heapq.heappush(data, 4)
print("Min Heap after push:", data)


# 弹出并返回最小元素
min_element = heapq.heappop(data)
print("Popped Min Element:", min_element)
print("Min Heap after pop:", data)


# 替换堆中的最小元素
# 弹出并返回最小元素，然后将新元素推入堆
min_element_replaced = heapq.heapreplace(data, 6)
print("Popped and Replaced Min Element:", min_element_replaced)
print("Min Heap after replace:", data)



# 使用堆实现优先队列
# 优先队列是一种数据结构，其元素具有优先级，可以用最小堆来实现。
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

#1
pq = PriorityQueue()
pq.push(2, 12)
pq.push(3, 14)
pq.pop()



#2
pq = PriorityQueue()
pq.push('a', 12)
pq.push('cc', 15)

pq.pop()

# 3
q = PriorityQueue()
q.push('lenovo', 1)
q.push('Mac', 5)
q.push('ThinkPad', 2)
q.push('Surface', 3)

q.pop()
# Mac
q.pop()
# Surface

# 使用堆进行排序
# 堆排序是一种利用堆数据结构的排序算法。
nums = [1,14,20,28,5,16,21,22,28,17]
print(nums)
def heap_sort(arr):
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]

res = heap_sort(nums)

print(res)










































































































































