#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:55:37 2024

@author: jack
"""


# 采用 * 和 ** 拆包


# 使用一个星号
complex_list = [3, 4]
complex(*complex_list)

# 使用两个星号
complex_dict = {'real': 3, 'imag': 4}
complex(**complex_dict)


# 自定义函数中使用*args
# 利用*args
def multiply_all(*args):
    result = 1
    print(args)
    for num_idx in args:
        result *= num_idx
    return result


# 计算4个值的乘积
print(multiply_all(1, 2, 3, 4))

# 计算6个值的乘积
print(multiply_all(1, 2, 3, 4, 5, 6))







# 自定义函数中使用**kwargs
# 利用*kwargs
def multiply_all_2(**kwargs):
    result = 1
    print(type(kwargs))
    # 循环dict()
    for key, value in kwargs.items():
        print("The value of {} is {}".format(key, value))
        result *= value

    return result

# 计算3个key-value pairs中值的乘积
print(multiply_all_2(A = 1, B = 2, C = 3))

# 计算4个key-value pairs中值的乘积
print(multiply_all_2(A = 1, B = 2, C = 3, D = 4))


# 自定义函数中混合使用args和*kwargs
import statistics
# 混合 *args, **kwargs
def calc_stats(operation, *args, **kwargs):
    result = 0
    # 计算标准差
    if operation == "stdev":
        # 总体标准差
        if "TYPE" in kwargs and kwargs["TYPE"] == 'population':
            result = statistics.pstdev(args)
        # 样本标准差
        elif "TYPE" in kwargs and kwargs["TYPE"] == 'sample':
            result = statistics.stdev(args)
        else:
            raise ValueError('TYPE, either population or sample')
    # 计算方差
    elif operation == "var":
        # 总体方差
        if "TYPE" in kwargs and kwargs["TYPE"] == 'population':
            result = statistics.pvariance(args)
        # 样本方差
        elif "TYPE" in kwargs and kwargs["TYPE"] == 'sample':
            result = statistics.variance(args)
        else:
            raise ValueError('TYPE, either population or sample')
    else:
        print("Unsupported operation")
        return None
    # 保留小数位
    if "ROUND" in kwargs:
        result = round(result, kwargs["ROUND"])
    return result


# 计算总体标准差
calc_stats("stdev", 1, 2, 3, 4, 5, 6,
            TYPE = 'population', ROUND = 3)


# 计算样本标准差
calc_stats("stdev", 1, 2, 3, 4, 5, 6, TYPE = 'sample')



# 计算总体方差
calc_stats("var", 1, 2, 3, 4, 5, 6,
           TYPE = 'population', ROUND = 4)

# 计算样本方差
calc_stats("var", 1, 2, 3, 4, 5, 6, TYPE = 'sample')












































































































































































































































































































































































































































































































































































































































































