#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 21:02:33 2022

@author: jack
"""

from cProfile import Profile

def runRe():
    import re
    re.compile("aaa|bbb")

prof = Profile()
prof.enable()
runRe()
prof.create_stats()
prof.print_stats()