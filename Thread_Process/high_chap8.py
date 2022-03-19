#########################################################################
# File Name: high_chap8.py
# Author: chenjunjie
# mail: 2716705056@qq.com
# Created Time: 2018.08.18
#########################################################################
#!/usr/bin/env python3
#! -*- coding: utf-8 -*-
import requests
import string
import random
import time
def generate_urls(base_url,num_urls):
    for i in range(num_urls):
        yield base_url+"".join(random.sample(string.ascii_lowercase,10))

def run_experiment(base_url,num_iter=500):
    reponse_size = 0
    for url in generate_urls(base_url,num_iter):
        reponse = requests.get(url)
        reponse_size += len(reponse.text)
    return reponse_size

def main():
    delay = 10
    num_iter = 100
    base_url = "http://127.0.0.1:8080/add?name=serial&delay={}&".format(delay)

    start = time.time()
    result = run_experiment(base_url,num_iter)
    end = time.time()
    print("Result : {}, Time : {}".format(result,end-start))


if __name__=='__main__':
    main()
