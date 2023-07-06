#!/usr/bin/env python3.6
#-*-coding=utf-8-*-


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
print(math.log(125, 5)) # 以5为底数
print(np.log(125)/ np.log(5)) #numpy在使用任意底数计算log的时候有点麻烦，需要log除法的公式转换一下
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
