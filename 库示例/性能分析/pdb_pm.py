#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:45:42 2022

@author: jack
"""

#!/usr/bin/env python

import pdb
import sys

def test_debugger(some_int):
    print( "start some_int>>", some_int)
    return_int = 10 / some_int
    print ("end some_int>>", some_int)
    return return_int

def do_debugger(type, value, tb):
    pdb.pm()

if __name__ == "__main__":
    sys.excepthook = do_debugger
    test_debugger(0)