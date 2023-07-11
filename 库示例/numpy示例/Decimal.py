#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:04:06 2023

@author: jack
"""

import numpy as np
import decimal
from scipy.special import comb
from decimal import Decimal, getcontext, setcontext, Context, ROUND_HALF_UP, ROUND_HALF_DOWN

# 1、其他博文说明的用法：设置 decimal.getcontext().prec = 3
decimal.getcontext().prec = 3

# getcontext().prec = 2
# 也可以直接设置decimal环境变量
# mycontext = Context(prec=2, rounding=ROUND_HALF_DOWN)
# setcontext(mycontext)

print(decimal.Decimal('2.3') / decimal.Decimal('3'))  # 结果：0.767
print(decimal.Decimal('1.22222') * decimal.Decimal('0.01')) # 结果：0.0122
print(decimal.Decimal('1') / decimal.Decimal('3'))  # 结果：0.333
# 很明显，此种用法，decimal.getcontext().prec = 3 并不是保留3位小数，而是保留3位有效数字

#2、用法二： round
print(round(1.626, 2)) # 结果：1.63
print(round(1.625, 2)) # 结果：1.62
print(round(1.624, 2)) # 结果：1.62
# 此种用法，在四舍五入的处理，并不能按照预想的 遇5进1.


# 3、用法三：使用quantize，保留两位小数，则 设置为0.00

decimal.getcontext().rounding = decimal.ROUND_HALF_UP # 表示四舍五入
print(decimal.Decimal("1.625").quantize(decimal.Decimal('0.00')))




from decimal import Decimal, ROUND_HALF_UP
num = Decimal(12314.424234).quantize(Decimal('0.00'), ROUND_HALF_UP)
