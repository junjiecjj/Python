#!/usr/bin/env python3.6
#-*-coding=utf-8-*-


str1 = '汗'
print (repr(str1))

str2 = u'汗'
print (repr(str2))
str3 = str2.encode('utf-8')
str4 = str2.encode('gbk')
print (repr(str3))
print (repr(str4))
str5 = str3.decode('utf-8') 
print( repr(str5))
str6 = str4.decode('gbk') 
print( repr(str6))