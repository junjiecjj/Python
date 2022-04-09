#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:14:22 2022

@author: jack
"""



from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


encoded_input = tokenizer("我是一句话")
print(encoded_input)



tokenizer.decode(encoded_input["input_ids"])


encoded_input = tokenizer("您贵姓?", "免贵姓李")
print(encoded_input)




















































































