#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:44:19 2024

@author: jack
"""




# Python 两个list交集、并集、差集的方法

############ 二、交集的方法
# 方法一：使用set类型的intersection()方法
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

set1 = set(list1)
set2 = set(list2)

intersection = set1.intersection(set2)
result = list(intersection)

print(result)

# 方法二：使用列表推导式
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

intersection = [x for x in list1 if x in list2]

print(intersection)


######## 三、并集的方法
# 方法一：使用set类型的union()方法
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

set1 = set(list1)
set2 = set(list2)

union = set1.union(set2)
result = list(union)

print(result)

# 方法二：使用+操作符和列表推导式
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

union = list1 + [x for x in list2 if x not in list1]

print(union)

########## 四、差集的方法
# 方法一：使用set类型的difference()方法
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

set1 = set(list1)
set2 = set(list2)

difference = set1.difference(set2)
result = list(difference)

print(result)

# 方法二：使用列表推导式
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

difference = [x for x in list1 if x not in list2]

print(difference)




import numpy as np

# numpy 中有以下类似 set 集合的操作：
# 差集 setdiff1d()，异或集 setxor1d()，并集 union1d()，交集 intersect1d()，以及判断是否存在的 in1d() 和 isin()


################>>>>>>>>>>>>>>>>>>>>> 是否存在
import numpy as np
test = np.array([0, 1, 2, 5, 0])
states = [0, 2]
mask = np.in1d(test, states)
print(mask)
# array([ True, False,  True, False,  True])
print(test[mask])
# array([0, 2, 0])
mask = np.in1d(test, states, invert=True)
print(mask)
# array([False,  True, False,  True, False])
print(test[mask])
# array([1, 5])

################>>>>>>>>>>>>>>>>>>>>> 是否存在
import numpy as np
element = 2*np.arange(4).reshape((2, 2))
print(element)
# [[0 2]
 # [4 6]]

test_elements = [1, 2, 4, 8]
mask = np.isin(element, test_elements)
print(mask)
# [[False  True]
 # [ True False]]

print(element[mask])
print(np.nonzero(mask))
# [2 4]
# (array([0, 1]), array([1, 0]))

mask = np.isin(element, test_elements, invert=True)
print(mask)
print(element[mask])
# [[ True False]
 # [False  True]]
# [0 6]

test_set = {1, 2, 4, 8}
print(np.isin(element, test_set))
# array([[False, False],
       # [False, False]])
print(np.isin(element, list(test_set)))
# [[False  True]
 # [ True False]]
################>>>>>>>>>>>>>>>>>>>>> 交集
import numpy as np
a = np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
print(a)
# array([1, 3])

x = np.array([1, 1, 2, 3, 4])
y = np.array([2, 1, 4, 6])
xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
print(xy, x_ind, y_ind)
# [1 2 4] [0 2 4] [1 0 2]
print(xy, x[x_ind], y[y_ind])
# (array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))

################>>>>>>>>>>>>>>>>>>>>> 差集
import numpy as np
a = np.array([1, 2, 3, 2, 4, 1])
b = np.array([3, 4, 5, 6])
print(np.setdiff1d(a, b))
# array([1, 2])


################>>>>>>>>>>>>>>>>>>>>> ，异或集
import numpy as np
a = np.array([1, 2, 3, 2, 4])
b = np.array([2, 3, 5, 7, 5])
print(np.setxor1d(a,b))
# array([1, 4, 5, 7])


################>>>>>>>>>>>>>>>>>>>>> 并集

import numpy as np
print(np.union1d([-1, 0, 1], [-2, 0, 2]))


from functools import reduce
print(reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2])))
# array([1, 2, 3, 4, 6])













