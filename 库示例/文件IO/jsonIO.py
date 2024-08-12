#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 16:56:52 2024

@author: jack
"""




import json
import numpy as np


data = {
    "name": "John",
    "age": 30,
    "city": "New York",
    # "array":np.random.randn(3,4), TypeError: Object of type ndarray is not JSON serializable
}

# 打开一个文件，将字典转储为JSON格式并保存到文件中
with open('data.json', 'w') as f:
    json.dump(data, f)

# 打开JSON文件并将其加载为字典
with open('data.json', 'r') as f:
    data1 = json.load(f)

# 输出转换后的字典
print(data1)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 有缩进：
from collections import defaultdict, OrderedDict
import json

video = defaultdict(list)
video["label"].append("haha")
video["data"].append(234)
video["score"].append(0.3)
video["label"].append("xixi")
video["data"].append(123)
video["score"].append(0.7)

test_dict = {
    'version': "1.0",
    'results': video,
    'explain': {
        'used': True,
        'details': "this is for josn test",
  }
}

# 字典内容写入json时，需要用json.dumps将字典转换为字符串，然后再写入。
json_str = json.dumps(test_dict, indent=4)
with open('test_data.json', 'w') as json_file:
    json_file.write(json_str)


# 打开JSON文件并将其加载为字典
with open('test_data.json', 'r') as f:
    test_dict1 = json.load(f)

# 输出转换后的字典
print(test_dict1)












