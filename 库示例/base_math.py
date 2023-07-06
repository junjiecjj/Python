#!/usr/bin/env python3.6
# -*-coding=utf-8-*-


import math   # 导入 math 模块
import string
import numpy as np


import math
import numpy as np
# 不写底数时默认以e为底
np.log(100)
math.log(100)
# 以2 e 10为底
print(np.log(100))
print(math.log(100))
print(np.log(np.e))
print(math.log(math.e))
print(np.log2(100))
print(math.log2(100))
print(np.log10(100))
print(math.log10(100))
# 任意底数
print(math.log(125, 5))  # 以5为底数
print(np.log(125) / np.log(5))  # numpy在使用任意底数计算log的时候有点麻烦，需要log除法的公式转换一下
# 结果：
# 4.605170185988092
# 4.605170185988092
# 1.0
# 1.0
# 6.643856189774724
# 6.643856189774724
# 2.0
# 2.0
# 3.0
# 3.0


# np.log 和math.log的底数是什么，默认都是e
# np.log()
# 一直分不清楚log到底是以什么为底，就写下这个作为备忘

# 看到没，是以e为底的，如果是其他的就logn

print('np.e:', np.e)
print('np.log([100,10000,10000]:', np.log( [100, 10000, 10000]))  # 输出结果是不是和你想象的不一样，请往下看
print('np.log10([100,10000,10000]:', np.log10([100, 10000, 10000]))  # 以10为底的对数
# np.log(x) 默认底数为自然数e
print('np.log([1,np.e,np.e**2]):', np.log([1, np.e, np.e**2]))
print('np.log2([2,2**2,2**3]):', np.log2([2, 2**2, 2**3]))  # 以2为底的对数


print("math.exp(-45.17) : ", math.exp(-45.17))
print("math.exp(100.12) : ", math.exp(100.12))
print("math.exp(100.72) : ", math.exp(100.72))
print("math.exp(119L) : ", math.exp(119))
print("math.exp(math.pi) : ", math.exp(math.pi))

# math.log()
# 但是，math的log又是怎么使用的，默认还是以e为底

# 一般用法是math.log(a,底数),其中只输入一个值，则是以e为底
math.log(10)  # 2.302585092994046
math.log(math.e)  # 1.0
math.log(100, 10)  # 2.0
math.log(10, 100)  # 0.5
print("math.log(100.12) : ", math.log(100.12))
print("math.log(100.72) : ", math.log(100.72))

print("math.log(math.pi) : ", math.log(math.pi))
# 设置底数
print("math.log(10,2) : ", math.log(10, 2))


print(f"np.exp(2) = {np.exp(2)}")
print(f"2**3 = {2**3}")
