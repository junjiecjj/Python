#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 22:17:15 2022

@author: jack
"""



def func():
  print("Hi, I am function")

class TestClass(object):
  def func(self):
    print("Hi, I am class")


if __name__ == "__main__":
  func()
  TestClass().func()

