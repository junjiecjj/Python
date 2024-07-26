
"""

逻辑回归，一个强大算法模型！
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247483792&idx=1&sn=a7d4c8b718d3ae979569aa1fd161e70b&chksm=c0e5db56f7925240579336264b761e8d35faa1b0f6549a7a5d9adc6db498d3cdf8e9162c1770&mpshare=1&scene=1&srcid=0726br4rnsI37hbe7jmfVkY6&sharer_shareinfo=b04350edfa67ac3fb84559d456879741&sharer_shareinfo_first=b04350edfa67ac3fb84559d456879741&exportkey=n_ChQIAhIQ%2BfV4SY5T2H6IHFbYowLoFhKfAgIE97dBBAEAAAAAAGyiMHuQBmcAAAAOpnltbLcz9gKNyK89dVj0%2Bkt3TDhNtFBSKM2eGzpzFc8h3%2B0zRakVT1S81gmNxt5VgDUzF3KEcjBRSH3TxnJagcDIg8k2RiQvy3YcC9VuZvJ7UGOcI7eh%2B6U16cCjHqGpg8KE1Y6cbTKz6WU3p2YQ4W1lMpMN90XK5mqKgYA0l7rGBCz5hKQTtX1DYte1%2BIxBpQ%2BCs%2F40qtGNPBkG3DTPqaaLQFZG4ZHOiePynzjPVsa8XN5jzvWoF0nRGTdXTNTBTouZR%2Bg1kJy%2FKsH%2BDXhPHBBUa76myBPjuNkt%2F4%2BrYbsVOShZ59jRuTHR8N7xFD6NBTHKbpa5NcbKi2hFQClBipDmRDmyYaKK&acctmode=0&pass_ticket=HhCVQdbJnetdzTglynwVoxMHpNLj99qTW4wcHxvVnwBxaSRmD6CfdO%2Bl2rM%2BA00J&wx_header=0#rd



"""


import numpy as np
import matplotlib.pyplot as plt

# 定义逻辑函数
def logistic_function(x):
    return 1 / (1 + np.exp(-x))

# 生成一些随机数据
x = np.linspace(-6, 6, 100)
y = logistic_function(x)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Logistic Function', color='blue')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Logistic Regression Function')
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()
























