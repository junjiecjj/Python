#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 16:55:24 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0MjUxMzg3OQ==&mid=2247494482&idx=1&sn=e46b9e07fb81ac943796a55935e19abc&chksm=c333bf9d233e4c59e4f336ae440fa81f84dcac47608b192fd752808846f08839b9c6baeed159&mpshare=1&scene=1&srcid=0725PSHbM6Bx8cGVZ6y007t2&sharer_shareinfo=ebc3cbfe37b7cd21216f699593b92c76&sharer_shareinfo_first=ebc3cbfe37b7cd21216f699593b92c76&exportkey=n_ChQIAhIQFDaTPwzWwxIEP6wbIVarQRKfAgIE97dBBAEAAAAAACBBOi8n3MkAAAAOpnltbLcz9gKNyK89dVj0c1gdNa0%2BR8b05%2FbVyEHOjMoJcx3Q3EYQvKh67J0VC57eMDdwXYtNeDHvWVZN%2FJhnLrLE0JiQctZFm9Iemsap4C7CSZW9PwYA1QShxUaJ3bENeBeQRI0T93JKvAfi%2Bbw726ECK2QxtOY5teCYn1HX3zeEAn%2B14V9rOm5eznybC53F1c84D%2Bgk9CF10W8Q7EG6uciW95sf5pk8k1T5YxaMBSUUvNxOG84g%2FStfnkVLGGjecCmxyKIR1CD33pmojSsWAX6XzyIlmzsOV2Zn3fDXXd0c1Ru5q%2FSl5BA0LNOZZZ%2FB3063El0doMbfJml%2F5bGWS91ayjPhcPt7&acctmode=0&pass_ticket=vSXfbaHXcOSHmn9Wa15vWmHsPu0El06AFNzDYsETU5YM1ZUy6obavaniINnBi%2Foj&wx_header=0#rd


"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from collections import Counter
import math

# 1. 加载“垃圾”和“正常”邮件：这里只拿 sci.space 当“正常”，talk.politics.* 当“垃圾”
cats = ['sci.space', 'talk.politics.misc']
data = fetch_20newsgroups(subset='all', categories=cats, remove=('headers','footers','quotes'))
texts, labels = data.data, data.target
# labels: 0->sci.space (正常), 1->talk.politics.misc (垃圾)

# 2. 简易分词和去停用（极简处理）
def tokenize(text):
    words = text.lower().split()
    # 去掉非字母开头的词，长度<2 的词
    return [w for w in words if w.isalpha() and len(w)>2]

tokenized = [tokenize(t) for t in texts]

# 3. 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(tokenized, labels, test_size=0.3, random_state=42)

# 4. 统计先验和条件概率所需的计数
# 4.1 先验
n_spam = sum(1 for y in y_train if y==1)
n_ham  = sum(1 for y in y_train if y==0)
total  = len(y_train)
p_spam = n_spam / total
p_ham  = n_ham  / total

# 4.2 条件概率计数
alpha = 1.0  # 拉普拉斯平滑参数
spam_words = Counter()
ham_words  = Counter()
for words, label in zip(X_train, y_train):
    if label==1:
        spam_words.update(words)
    else:
        ham_words.update(words)
vocab = set(spam_words.keys()) | set(ham_words.keys())
V = len(vocab)

# 4.3 计算每个词的 log 概率，存在 dict 里
log_p_w_spam = {}
log_p_w_ham  = {}
# 统计所有词总数
total_spam_words = sum(spam_words.values())
total_ham_words  = sum(ham_words.values())

for w in vocab:
    # P(w|spam) = (count_spam(w)+alpha) / (total_spam_words + alpha*V)
    log_p_w_spam[w] = math.log((spam_words[w] + alpha) / (total_spam_words + alpha*V))
    log_p_w_ham[w]  = math.log((ham_words[w]  + alpha) / (total_ham_words  + alpha*V))

# 5. 定义分类函数
def predict(words):
    # 初始化为 log 先验
    log_spam = math.log(p_spam)
    log_ham  = math.log(p_ham)
    for w in words:
        if w in vocab:
            log_spam += log_p_w_spam[w]
            log_ham  += log_p_w_ham[w]
        else:
            # 未登录词，概率 = alpha / (total + alpha*V)
            log_spam += math.log(alpha / (total_spam_words + alpha*V))
            log_ham  += math.log(alpha / (total_ham_words  + alpha*V))
    # 返回概率更大的一类
    return 1 if log_spam > log_ham else 0

# 6. 在测试集上评估
y_pred = [predict(words) for words in X_test]
acc = sum(1 for i in range(len(y_pred)) if y_pred[i]==y_test[i]) / len(y_test)
print(f"朴素贝叶斯分类准确率: {acc:.4f}")
