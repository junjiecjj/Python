#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:14:22 2022

@author: jack
"""


#Tokenizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


encoded_input = tokenizer("我是一句话")
print(encoded_input)



tokenizer.decode(encoded_input["input_ids"])


encoded_input = tokenizer("您贵姓?", "免贵姓李")
print(encoded_input)



batch_sentences = ["我是一句话", "我是另一句话", "我是最后一句话"]
batch = tokenizer(batch_sentences, padding=True, return_tensors="pt")
print(batch)



#Model

from transformers import BertModel
model = BertModel.from_pretrained("bert-base-chinese")
bert_output = model(input_ids=batch['input_ids'])

print(f"len(bert_output) = {len(bert_output)}")

print(f"bert_output[0].shape = {bert_output[0].shape}")
print(f"bert_output[1].shape = {bert_output[1].shape}")








#下游任务


























































