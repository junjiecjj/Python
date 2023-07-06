#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://www.jianshu.com/p/12a8207149b0



from memory_profiler import profile


import objgraph



_cache = []
class OBJ(object):
    pass

def func_to_leak():
    o  = OBJ()
    _cache.append(o)
# do something with o, then remove it from _cache 
    if True: # this seem ugly, but it always exists
        return 
    _cache.remove(o)

if __name__ == '__main__':
    objgraph.show_growth()
    try:
        func_to_leak()
    except:
        pass
    print('after call func_to_leak')
    objgraph.show_growth()
    #objgraph.show_backrefs(objgraph.by_type('OBJ')[0], max_depth = 10, filename = 'obj.png')

    objgraph.show_chain(
        objgraph.find_backref_chain(
            objgraph.by_type('OBJ')[0],
            objgraph.is_proper_module
            ), filename='obj_chain.png'
        )

print(f"================================== 1 =====================================")

#objgraph.show_most_common_types(25)

print(f"================================== 2 =====================================")






