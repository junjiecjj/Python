# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:35:02 2021

@author: Parthe
"""

class BaseSolver(object):
    
    def __init__(self, list_of_variable_names):
        self.list_of_variable_names = list_of_variable_names
        for variable in list_of_variable_names:
            setattr(self,variable,[])
    
    def update(self, list_of_values):
        for value,variable in zip(list_of_values, self.list_of_variable_names):
            setattr(self,variable,eval('self.'+variable)+[value])