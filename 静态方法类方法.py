# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:45:01 2017

@author: 科
"""

class Myclass(object):
    val1='value1'
    def __init__(self):
        self.val2='value2'
        self.val3='value3'

    '''普通方法，也叫类的实例方法（也就是将类实例化后可以调用的方法），
    必须传入self参数，最常用的方法；可以由实例调用，但是不能被类直接调用；
    可以访问实例变量，也可以访问类变量（在类变量前加self.val1）'''
    def normal_method(self):
        print('val1: %s'% self.val1)
        print('val2: %s'% self.val2)
        print('val3: %s'% self.val3)
        print('val1: %s val2: %s val3: %s.'% (self.val1,self.val2,self.val3))
        print("******分割线1**********\n")

    '''静态方法，可以由实例和类直接调用，不能访问类变量和实例变量，
    但可以传递其他参数'''
    @staticmethod
    def static_method(name):
        print(name)
        print('不能访问val1、val2、val3')
        #print('val1: %s'% self.val1)
        #print('val2: %s'% self.val2)
        #print('val3: %s'% self.val3)
        #print('val1: %s val2: %s val3: %s'% (cls.val1,self.val2,self.val3))
        print("***********分割线2**************\n")

    '''类方法，也叫成员方法，必须传入cls参数，可以访问类变量，但不能
    访问实例变量，可以由实例和类直接调用'''
    @classmethod
    def class_method(cls):
        print('val1: %s'% cls.val1)
        #print('val2: %s'% cls.val2)
        #print('val3: %s'% cls.val3)
        #print('val1: %s\n val2: %s\n val3: %s\n'% (cls.val1,cls.val2,cls.val3))
        print("不能访问val2,val3")
        print("*********分割线3**********\n")

    @classmethod
    def False_class_method(cls):
        print('val1: %s'% self.val1)
        print('val2: %s'% self.val2)
        print('val3: %s'% self.val3)
        print('val1: %s val2: %s val3: %s'% (self.val1,self.val2,self.val3))
        print("********分割线***********\n")


my = Myclass()
my.normal_method()
my.static_method("chenjunjie")
my.class_method()

