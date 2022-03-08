#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:57:44 2022

https://zhuanlan.zhihu.com/p/435395340

@author: jack
"""

import os
import numpy as np
import matplotlib.pyplot as plt

H = np.array([[1,1,0,1,1,0,1,1,0],\
			  [0,1,1,0,1,1,0,1,1],\
			  [1,0,1,1,0,1,1,0,1]])


def ldpc_decode_data(h,delta):
	length = len(h[0])
	data = np.random.randint(low = 0, high= 2, size=length, dtype='int')
	encode_data = np.zeros(length + len(h),dtype='float')
	encode_data[0:len(data)] = data

	for i in range(len(h)):
		tmp = 0
		for j in range(length):
			tmp ^= (h[i][j]&data[j])
		encode_data[len(data) + i] = tmp

	encode_data -= 0.5
	encode_data *= -2

	#numpy.random. normal ( loc=0.0 , scale=1.0 , size=None )loc 均值，scale 标准差，size大小
	noise_data = np.random.normal(loc=0.0 ,scale=delta,size=len(encode_data))
	noise_data = encode_data + noise_data

	return data,noise_data


def bit_judge(data_in):

	bits = np.zeros(len(data_in),dtype = 'int')
	for i in range(len(bits)):
		if data_in[i] < 0 :
			bits[i] = 1

	return bits


def ldpc_decode(h,decode_data, iter_num = 5, alpha = 0.75):

	LPn = np.zeros(h.shape,dtype = 'float')
	Lqmn = np.zeros(h.shape,dtype = 'float')
	Lrmn = np.zeros(h.shape,dtype = 'float')
	LQn = np.zeros(len(Lqmn[0]),dtype = 'float')

	check_data = decode_data[len(Lqmn[0]):]

	#print(decode_data)
	#print(check_data)

	for row in range(len(Lqmn)):
		for col in range(len(Lqmn[0])):
			if h[row][col] == 1:
				LPn[row][col] = decode_data[col]
				Lqmn[row][col] = decode_data[col]

	for iter in range(iter_num):
		for row in range(len(Lqmn)):
			for col in range(len(Lqmn[0])):
				if h[row][col] == 1:
					sign = 1.0
					if check_data[row] < 0:
						sign = -1.0
					min_data = abs(check_data[row])

					for col_idx in range(len(Lqmn[0])):
						if h[row][col_idx] == 1 and col_idx != col:
							if Lqmn[row][col_idx] < 0:
								sign *= -1

							if abs(Lqmn[row][col_idx]) < min_data:
								min_data = abs(Lqmn[row][col_idx])

					Lrmn[row][col] = min_data * alpha * sign
				else:
					Lrmn[row][col] = 0.0

		#print ("Lrmn:",Lrmn)

		for row in range(len(Lrmn)):
			for col in range(len(Lrmn[0])):
				if h[row][col] == 1:
					sum_tmp = 0
					for row_idx in range(len(Lrmn)):
						if row_idx != row:
							sum_tmp += Lrmn[row_idx][col]

					Lqmn[row][col] = LPn[row][col] + sum_tmp

		#print ("Lqmn:",Lqmn)

	for col in range(len(LQn)):
		sum_tmp = 0.0
		for row in range(len(Lqmn)):
			sum_tmp += Lrmn[row][col]

		LQn[col] = sum_tmp

	#print ("LQnIn:",LQn)
	#print ("decode_data:",decode_data)
	LQn = LQn + decode_data[0:len(LQn)]
	#print ("LQnOut:",LQn)

	bits = bit_judge(LQn)

	return bits


def err_bit_count(bits0,bits1):
	number = 0
	for i in range(len(bits0)):
		if bits0[i] != bits1[i]:
			number += 1

	return number


def test_once(h,delta = 0.1):

	encode_data,decode_data = ldpc_decode_data(h,delta)
	decode_out = ldpc_decode(h,decode_data)

	#print ('encode_data:',encode_data)
	#print ('decode_out :',decode_out)

	hard_judge_bits = bit_judge(decode_data[0:len(h[0])])

	total_number = len(encode_data)
	err_bits_num_soft = err_bit_count(encode_data,decode_out)
	err_bits_num_hard = err_bit_count(encode_data,hard_judge_bits)

	return total_number, err_bits_num_soft, err_bits_num_hard

if __name__ == '__main__':

	sinr_start = -1.0
	sinr_end = 15
	sinr_step = 1
	loop_num = 1000
	log_num = int((sinr_end - sinr_start)/sinr_step)
	err_soft_log = np.zeros(log_num,dtype = 'float')
	err_hard_log = np.zeros(log_num,dtype = 'float')

	for i in range(log_num):
		total_acc = 0.0
		err_soft_acc = 0.0
		err_hard_acc = 0.0
		sinr = sinr_start + sinr_step * i
		delta = (10**(-sinr/10)) ** 0.5
		for _ in range(loop_num):
			total_number, err_bits_num_soft, err_bits_num_hard =test_once(H,delta = delta)
			total_acc += total_number
			err_soft_acc += err_bits_num_soft
			err_hard_acc += err_bits_num_hard

		err_soft_log[i] = err_soft_acc/total_acc
		err_hard_log[i] = err_hard_acc/total_acc

	print (err_soft_log)
	print (err_hard_log)

	x = np.arange(sinr_start, sinr_end, sinr_step)
	plt.plot(x, err_soft_log)
	plt.plot(x, err_hard_log)
	plt.yscale('log')
	plt.show()