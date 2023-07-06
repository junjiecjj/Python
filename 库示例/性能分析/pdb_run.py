#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:56:19 2022

@author: jack
"""

#!/usr/bin/env python

import pdb

def test_debugger(some_int):
    print ("start some_int>>", some_int)
    return_int = 10 / some_int
    print( "end some_int>>", some_int)
    return return_int

if __name__ == "__main__":
    pdb.run("test_debugger(0)")