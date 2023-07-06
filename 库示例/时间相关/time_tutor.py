#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:23:26 2022

@author: jack

https://www.cnblogs.com/tkqasn/p/6001134.html

 time模块中时间表现的格式主要有三种：

　　a、timestamp时间戳，时间戳表示的是从1970年1月1日00:00:00开始按秒计算的偏移量

　　b、struct_time时间元组，共有九个元素组。

　　c、format time 格式化时间，已格式化的结构使时间更具可读性。包括自定义格式和固定格式。



                                    ***********************
                                    *    struct time      *
                                    ***********************
                                   *                       *
                                  *                         *
                                *                            *
                              *                               *
                strftime \/ *  strptime ^            mktime \/ * localtime ^
                          *                                     *
                        *                                        *
                      *                                           *
**********************                                             *****************
*   Format string    *                                             *   Timestamp   *
**********************                                             *****************



                                 *******************************
                                 *   %a %b %d %H:%M:%S %Y      *
                                 *******************************
                                   *                       *
                                  *                         *
                                *                            *
                              *                               *
                  asctime ^ *                                  * ctime ^
                          *                                     *
                        *                                        *
                      *                                           *
**********************                                             *****************
*   Struct time      *                                             *   Timestamp   *
**********************                                             *****************






struct_time元组元素结构
属性                            值
tm_year（年）                  比如2011
tm_mon（月）                   1 - 12
tm_mday（日）                  1 - 31
tm_hour（时）                  0 - 23
tm_min（分）                   0 - 59
tm_sec（秒）                   0 - 61
tm_wday（weekday）             0 - 6（0表示周日）
tm_yday（一年中的第几天）        1 - 366
tm_isdst（是否是夏令时）        默认为-1

format time结构化表示
格式	        含义
%a      	本地（locale）简化星期名称
%A      	本地完整星期名称
%b      	本地简化月份名称
%B      	本地完整月份名称
%c      	本地相应的日期和时间表示
%d      	一个月中的第几天（01 - 31）
%H      	一天中的第几个小时（24小时制，00 - 23）
%I      	第几个小时（12小时制，01 - 12）
%j      	一年中的第几天（001 - 366）
%m      	月份（01 - 12）
%M      	分钟数（00 - 59）
%p      	本地am或者pm的相应符
%S      	秒（01 - 61）
%U      	一年中的星期数。（00 - 53星期天是一个星期的开始。）第一个星期天之前的所有天数都放在第0周。
%w      	一个星期中的第几天（0 - 6，0是星期天）
%W      	和%U基本相同，不同的是%W以星期一为一个星期的开始。
%x      	本地相应日期
%X      	本地相应时间
%y      	去掉世纪的年份（00 - 99）
%Y      	完整的年份
%Z      	时区的名字（如果不存在为空字符）
%%      	‘%’字符

说明
python中时间日期格式化符号：
%y 两位数的年份表示（00-99）
%Y 四位数的年份表示（000-9999）
%m 月份（01-12）
%d 月内中的一天（0-31）
%H 24小时制小时数（0-23）
%I 12小时制小时数（01-12）
%M 分钟数（00-59）
%S 秒（00-59）
%a 本地简化星期名称
%A 本地完整星期名称
%b 本地简化的月份名称
%B 本地完整的月份名称
%c 本地相应的日期表示和时间表示
%j 年内的一天（001-366）
%p 本地A.M.或P.M.的等价符
%U 一年中的星期数（00-53）星期天为星期的开始
%w 星期（0-6），星期天为 0，星期一为 1，以此类推。
%W 一年中的星期数（00-53）星期一为星期的开始
%x 本地相应的日期表示
%X 本地相应的时间表示
%Z 当前时区的名称
%% %号本身

"""






"""
Python time.time()
time()函数返回自纪元以来经过的秒数。
对于Unix系统，January 1, 1970, 00:00:00在UTC是历元（其中，时间开始点）。


Python time.ctime()
time.ctime()以历元以来的秒为参数，返回一个表示本地时间的字符串。



Python time.sleep()
sleep()函数在给定的秒数内暂停（延迟）当前线程的执行。



time.struct_time类
时间模块中的几个函数（例如gmtime()，asctime()等）将time.struct_time对象作为参数或将其返回。
这是一个time.struct_time对象的实例。
time.struct_time(tm_year=2018, tm_mon=12, tm_mday=27, 
                    tm_hour=6, tm_min=35, tm_sec=17, 
                    tm_wday=3, tm_yday=361, tm_isdst=0)

索引   属性	         属性值
0	tm_year	          0000，....，2018，...，9999
1	tm_mon	          1，2，...，12
2	tm_mday	          1，2，...，31
3	tm_hour	          0，1，...，23
4	tm_min	          0，1，...，59
5	tm_sec	          0，1，...，61
6	tm_wday	          0, 1, ..., 6; Monday 为 0
7	tm_yday	          1，2，...，366
8	tm_isdst	      0、1或-1
可以使用索引和属性访问time.struct_time对象的值（元素）。



Python time.localtime()
localtime()函数将自epoch以来经过的秒数作为参数，并以localtime返回struct_time。

"""

print("-"*70)
print("   https://www.cnblogs.com/tkqasn/p/6001134.html  start")
print("-"*70)

import time
# time()函数返回自纪元以来经过的秒数。
seconds = time.time()
print("Seconds since epoch =", seconds)



import time
localtime = time.localtime(time.time())
print ("本地时间为 :", localtime)

import time
localtime = time.localtime()
print ("本地时间为 :", localtime)


import time
localtime = time.asctime( time.localtime() )
print ("本地时间为 :", localtime)

import time
localtime = time.asctime( time.localtime(time.time()) )
print ("本地时间为 :", localtime)



import time
 
# 格式化成2016-03-20 11:45:39形式
print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
 
# 格式化成Sat Mar 28 22:24:24 2016形式
print (time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) )
  
# 将格式字符串转换为时间戳
a = "Sat Mar 28 22:24:24 2016"
print( time.mktime(time.strptime(a,"%a %b %d %H:%M:%S %Y")))




import time

# 生成timestamp
print("time.time() = {}\n".format(time.time()))
# 1477471508.05


#struct_time to timestamp
print("time.mktime(time.localtime()) = {}\n".format(time.mktime(time.localtime())))


print("time.mktime(time.localtime(time.time())) = {}\n".format(time.mktime(time.localtime(time.time()))))


#生成struct_time
# timestamp to struct_time 本地时间
print("time.localtime() = \n{}\n".format(time.localtime()))
print("time.localtime(time.time()) = \n{}\n".format(time.localtime(time.time())))
# time.struct_time(tm_year=2016, tm_mon=10, tm_mday=26, tm_hour=16, tm_min=45, tm_sec=8, tm_wday=2, tm_yday=300, tm_isdst=0)

# timestamp to struct_time 格林威治时间
print("time.gmtime() = \n{}\n".format(time.gmtime()))
print("time.gmtime(time.time()) = \n{}\n".format(time.gmtime(time.time())))
# time.struct_time(tm_year=2016, tm_mon=10, tm_mday=26, tm_hour=8, tm_min=45, tm_sec=8, tm_wday=2, tm_yday=300, tm_isdst=0)

#format_time to struct_time
print("time.strptime('2011-05-05 16:37:06', '%Y-%m-%d %X') = \n{}\n".format(time.strptime('2011-05-05 16:37:06', '%Y-%m-%d %X')))
# time.struct_time(tm_year=2011, tm_mon=5, tm_mday=5, tm_hour=16, tm_min=37, tm_sec=6, tm_wday=3, tm_yday=125, tm_isdst=-1)

#生成format_time
#struct_time to format_time
print("time.strftime(\"%Y-%m-%d %X\") = \n{}\n".format(time.strftime("%Y-%m-%d %X")))
print("time.strftime(\"%Y-%m-%d %X\",time.localtime()) = \n{}\n".format(time.strftime("%Y-%m-%d %X",time.localtime())))
# 2016-10-26 16:48:41


#生成固定格式的时间表示格式
print("time.asctime(time.localtime()) = {}\n".format(time.asctime(time.localtime())))
print("time.ctime(time.time()) = {}\n".format(time.ctime(time.time())))
# Wed Oct 26 16:45:08 2016



print("-"*70)
print("https://www.runoob.com/python/python-date-time.html  start")
print("-"*70)
import calendar
 
cal = calendar.month(2016, 1)
print ("以下输出2016年1月份的日历:\n{}".format(cal))



"""
Python time time() 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。
语法
time()方法语法：
time.time()
参数:Nan。
返回值:返回当前时间的时间戳（1970纪元后经过的浮点秒数）。
"""
import time
print( "time.time(): %f " %  time.time())
print( time.localtime( time.time() ))
print( time.asctime( time.localtime(time.time()) ))


"""
Python time strftime() 函数用于格式化时间，返回以可读字符串表示的当地时间，格式由参数 format 决定。

语法
strftime()方法语法：

time.strftime(format[, t])
参数
format -- 格式字符串。
t -- 可选的参数 t 是一个 struct_time 对象。
返回值
返回以可读字符串表示的当地时间。

说明
python中时间日期格式化符号：

%y 两位数的年份表示（00-99）
%Y 四位数的年份表示（000-9999）
%m 月份（01-12）
%d 月内中的一天（0-31）
%H 24小时制小时数（0-23）
%I 12小时制小时数（01-12）
%M 分钟数（00=59）
%S 秒（00-59）
%a 本地简化星期名称
%A 本地完整星期名称
%b 本地简化的月份名称
%B 本地完整的月份名称
%c 本地相应的日期表示和时间表示
%j 年内的一天（001-366）
%p 本地A.M.或P.M.的等价符
%U 一年中的星期数（00-53）星期天为星期的开始
%w 星期（0-6），星期天为星期的开始
%W 一年中的星期数（00-53）星期一为星期的开始
%x 本地相应的日期表示
%X 本地相应的时间表示
%Z 当前时区的名称
%% %号本身
"""
#生成format_time
#struct_time to format_time
print("time.strftime(\"%Y-%m-%d %X\") = \n{}\n".format(time.strftime("%Y-%m-%d %X")))
print("time.strftime(\"%Y-%m-%d %X\",time.localtime()) = \n{}\n".format(time.strftime("%Y-%m-%d %X",time.localtime())))
#   2022-03-26 16:53:35





"""
Python time strptime() 函数根据指定的格式把一个时间字符串解析为时间元组。

语法
strptime()方法语法：

time.strptime(string[, format])
参数
string -- 时间字符串。
format -- 格式化字符串。
返回值
返回struct_time对象。
"""
import time
 
struct_time = time.strptime("30 Nov 00", "%d %b %y")
print ("返回的元组: {}\n".format(struct_time))





"""
Python time sleep() 函数推迟调用线程的运行，可通过参数secs指秒数，表示进程挂起的时间。

语法
sleep()方法语法：

time.sleep(t)
参数
t -- 推迟执行的秒数。
返回值
该函数没有返回值。

"""
import time
 
print( "Start : %s" % time.ctime())
time.sleep( 5 )
print( "End : %s" % time.ctime())




"""
Python time mktime() 函数执行与gmtime(), localtime()相反的操作，它接收struct_time对象作为参数，返回用秒数来表示时间的浮点数。

如果输入的值不是一个合法的时间，将触发 OverflowError 或 ValueError。

语法
mktime()方法语法：

time.mktime(t)
参数: t -- 结构化的时间或者完整的9位元组元素。
返回值: 返回用秒数来表示时间的浮点数。
"""
import time

t = (2009, 2, 17, 17, 3, 38, 1, 48, 0)
secs = time.mktime( t )
print ("time.mktime(t) : %f" %  secs)
print ("asctime(localtime(secs)): %s" % time.asctime(time.localtime(secs)))


"""
描述
Python time localtime() 函数类似gmtime()，作用是格式化时间戳为本地的时间。 如果sec参数未输入，则以当前时间为转换标准。 DST (Daylight Savings Time) flag (-1, 0 or 1) 是否是夏令时。

语法
localtime()方法语法：

time.localtime([ sec ])
参数: sec -- 转换为time.struct_time类型的对象的秒数。
返回值:该函数没有任何返回值。
"""
import time
 
print ("time.localtime() = {}\n".format(time.localtime()))





"""
描述
Python time gmtime() 函数将一个时间戳转换为UTC时区（0时区）的struct_time，可选的参数sec表示从1970-1-1以来的秒数。其默认值为time.time()，函数返回time.struct_time类型的对象。（struct_time是在time模块中定义的表示时间的对象）。

语法
gmtime()方法语法：

time.gmtime([ sec ])
参数: sec -- 转换为time.struct_time类型的对象的秒数。
返回值: 该函数没有任何返回值。
"""

import time

print ("time.gmtime() = {}".format(time.gmtime()))




"""
Python time ctime() 函数把一个时间戳（按秒计算的浮点数）转化为time.asctime()的形式。 如果参数未给或者为None的时候，将会默认time.time()为参数。它的作用相当于 asctime(localtime(secs))。

语法
ctime()方法语法：

time.ctime([ sec ])
参数:sec -- 要转换为字符串时间的秒数。
返回值:该函数没有任何返回值。
"""


import time

print ("time.ctime() = {}".format(time.ctime()))




"""
Python 3.8 已移除 clock() 方法 可以使用 time.perf_counter() 或 time.process_time() 方法替代。

Python time clock() 函数以浮点数计算的秒数返回当前的CPU时间。用来衡量不同程序的耗时，比time.time()更有用。

这个需要注意，在不同的系统上含义不同。在UNIX系统上，它返回的是"进程时间"，它是用秒表示的浮点数（时间戳）。而在WINDOWS中，第一次调用，返回的是进程运行的实际时间。而第二次之后的调用是自第一次调用以后到现在的运行时间。（实际上是以WIN32上QueryPerformanceCounter()为基础，它比毫秒表示更为精确）

语法
clock()方法语法：

time.clock()
参数
NA。
返回值
该函数有两个功能，

在第一次调用的时候，返回的是程序运行的实际时间；

以第二次之后的调用，返回的是自第一次调用后,到这次调用的时间间隔

在win32系统下，这个函数返回的是真实时间（wall time），而在Unix/Linux下返回的是CPU时间。
"""




def procedure():
    time.sleep(2.5)

# measure process time
t0 = time.perf_counter()
procedure()
print (time.perf_counter() - t0, "seconds process time")

t0 = time.process_time()
procedure()
print (time.process_time() - t0, "seconds process time")


# measure wall time
t0 = time.time()
procedure()
print (time.time() - t0, "seconds wall time")




"""
Python time asctime() 函数接受时间元组并返回一个可读的形式为"Tue Dec 11 18:07:14 2008"（2008年12月11日 周二18时07分14秒）的24个字符的字符串。

语法
asctime()方法语法：

time.asctime([t]))
参数: t -- 9个元素的元组或者通过函数 gmtime() 或 localtime() 返回的时间值。
返回值: 返回一个可读的形式为"Tue Dec 11 18:07:14 2008"（2008年12月11日 周二18时07分14秒）的24个字符的字符串。
"""

import time

t = time.localtime()
print( "time.asctime(t): %s " % time.asctime(t))



print("-"*70)
print("https://www.runoob.com/python/python-date-time.html  end")
print("-"*70)





#time 模块包含很多函数来执行跟时间有关的函数。 尽管如此，通常我们会在此基础之上构造一个更高级的接口来模拟一个计时器。例如：
import time

class Timer:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()



def countdown(n):
    while n > 0:
        n -= 1

# Use 1: Explicit start/stop
t = Timer()
t.start()
countdown(1000000)
t.stop()
print(t.elapsed)


# Use 2: As a context manager
with t:
    countdown(1000000)

print(t.elapsed)

with Timer() as t2:
    countdown(1000000)
print(t2.elapsed)



#在计时中要考虑一个底层的时间函数问题。一般来说， 使用 time.time() 或 time.clock() 计算的时间精度因操作系统的不同会有所不同。 而使用 time.perf_counter() 函数可以确保使用系统上面最精确的计时器。

#上述代码中由 Timer 类记录的时间是钟表时间，并包含了所有休眠时间。 如果你只想计算该进程所花费的CPU时间，应该使用 time.process_time() 来代替：

t = Timer(time.process_time)
with t:
    countdown(1000000)
print(t.elapsed)
#time.perf_counter() 和 time.process_time() 都会返回小数形式的秒数时间。 实际的时间值没有任何意义，为了得到有意义的结果，你得执行两次函数然后计算它们的差值。




