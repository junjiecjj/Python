#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:42:53 2022

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
    try:
        test_debugger(0)
    except:
        import sys
        tb = sys.exc_info()[2]
        pdb.post_mortem(tb)