#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:32:33 2022

@author: jack

datetime模块：
	1. 模块定义了两个常量：
		datetime.MINYEAR
		datetime.MAXYEAR
		这两个常量分别表示 datetime 所能表示的最小、最大年份。其中，MINYEAR = 1，MAXYEAR = 9999。
	2.
		datetime 模块定义了下面这几个类：

		datetime.date：表示日期的类。常用的属性有year, month, day；
		datetime.time：表示时间的类。常用的属性有hour, minute, second, microsecond；
		datetime.datetime：表示日期时间。
		datetime.timedelta：表示时间间隔，即两个时间点之间的长度。
		datetime.tzinfo：与时区有关的相关信息。
1）date 类
		date对象格式：datetime.date(2017, 12, 31)
		返回一个date对象：
			date.max、date.min：date对象所能表示的最大、最小日期；
			date.resolution：date对象表示日期的最小单位。这里是天。
			date.today()：返回一个表示当前本地日期的 date 对象；
			date.fromtimestamp(timestamp)：根据给定的时间戮，返回一个 date 对象；
			datetime.fromordinal(ordinal)：将Gregorian日历时间转换为date对象；（Gregorian Calendar：一种日历表示方法，类似于我国的农历，西方国家使用比较多，此处不详细展开讨论。）
		date对象的属性：
			date.year、date.month、date.day：年、月、日；
			date.replace(year, month, day)：生成一个新的日期对象，用参数指定的年，月，日代替原有对象中的属性。（原有对象仍保持不变）
			date.timetuple()：返回日期对应的time.struct_time对象；
			date.toordinal()：返回日期对应的Gregorian Calendar日期；
			date.weekday()：返回weekday，如果是星期一，返回0；如果是星期2，返回1，以此类推；
			data.isoweekday()：返回weekday，如果是星期一，返回1；如果是星期2，返回2，以此类推；
			date.isocalendar()：返回格式如(year，month，day)的元组；
			date.isoformat()：返回格式如'YYYY-MM-DD'的字符串；
			date.strftime(fmt)：自定义格式化字符串。
		date加减和比较：
			date2 = date1 + timedelta：
			日期加上一个间隔，返回一个新的日期对象
			date2 = date1 - timedelta：
			日期隔去间隔，返回一个新的日期对象
			timedelta = date1 - date2：
			两个日期相减，返回一个时间间隔对象
			date1 < date2：
			两个日期进行比较。
2）time类
		time对象格式：datetime.time(23, 59, 59, 999999)
		time对象的属性：
			time.hour、time.minute、time.second、time.microsecond：时、分、秒、微秒；
			time.tzinfo：时区信息；
			time.replace([hour[, minute[, second[, microsecond[, tzinfo]]]]])：创建一个新的时间对象，用参数指定的时、分、秒、微秒代替原有对象中的属性（原有对象仍保持不变）；
			time.isoformat()：返回型如”HH:MM:SS”格式的字符串表示；
			time.strftime(fmt)：返回自定义格式化字符串。
3）datetime类
		datetime对象格式：datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)
		返回一个datetime：
			datetime.min、datetime.max：datetime所能表示的最小值与最大值；
			datetime.resolution：datetime最小单位；
			datetime.today()：返回一个表示当前本地时间的datetime对象；
			datetime.now([tz])：返回一个表示当前本地时间的datetime对象，如果提供了参数tz，则获取tz参数所指时区的本地时间；
			datetime.utcnow()：返回一个当前utc时间的datetime对象；
			datetime.fromtimestamp(timestamp[, tz])：根据时间戮创建一个datetime对象，参数tz指定时区信息；
			datetime.utcfromtimestamp(timestamp)：根据时间戮创建一个datetime对象；
			datetime.combine(date, time)：根据date和time，创建一个datetime对象；
			datetime.strptime(date_string, format)：将格式字符串转换为datetime对象，data 与 time 类没有提供该方法。
		datetime的属性：
			datetime.year、month、day、hour、minute、second、microsecond、tzinfo：
			datetime.date()：获取date对象；
			datetime.time()：获取time对象；
			datetime.replace([year[, month[, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]]]]])：
			**datetime.timetuple() **
			**datetime.utctimetuple() **
			datetime.toordinal()
			datetime.weekday()
			datetime.isocalendar()
			datetime.isoformat([sep])
			datetime.ctime()：返回一个日期时间的C格式字符串，等效于time.ctime(time.mktime(dt.timetuple()))；
			datetime.strftime(format)
		datetime 对象同样可以进行比较，或者相减返回一个时间间隔对象，或者日期时间加上一个间隔返回一个新的日期时间对象。
4）timedelta 类
		timedelta对象格式：datetime.timedelta(天数, 秒数, 微秒)内部只存储days，seconds，microseconds，
		返回一个timedelta：
			timedelta.min：时间间隔对象的最小值,即 timedelta(-999999999).
			timedelta.max：时间间隔对象的最大值，即 timedelta(days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999).
			timedelta.resolution：时间间隔的最小单位,即 timedelta(microseconds=1).
		方法：
			timedelta.total_seconds()：计算时间间隔的总秒数
5）格式字符串
		datetime、date、time 都提供了 strftime() 方法，该方法接收一个格式字符串，输出日期时间的字符串表示。支持的转换格式如下：
		%a星期的简写。如 星期三为Web
		%A星期的全写。如 星期三为Wednesday
		%b月份的简写。如4月份为Apr
		%B月份的全写。如4月份为April
		%c: 日期时间的字符串表示。（如： 04/07/10 10:43:39）
		%d: 日在这个月中的天数（是这个月的第几天）
		%f: 微秒（范围[0,999999]）
		%H: 小时（24小时制，[0, 23]）
		%I: 小时（12小时制，[0, 11]）
		%j: 日在年中的天数 [001,366]（是当年的第几天）
		%m: 月份（[01,12]）
		%M: 分钟（[00,59]）
		%p: AM或者PM
		%S: 秒（范围为[00,61]，为什么不是[00, 59]，参考python手册~_~）
		%U: 周在当年的周数当年的第几周），星期天作为周的第一天
		%w: 今天在这周的天数，范围为[0, 6]，6表示星期天
		%W: 周在当年的周数（是当年的第几周），星期一作为周的第一天
		%x: 日期字符串（如：04/07/10）
		%X: 时间字符串（如：10:43:39）
		%y: 2个数字表示的年份
		%Y: 4个数字表示的年份
		%z: 与utc时间的间隔 （如果是本地时间，返回空字符串）
		%Z: 时区名称（如果是本地时间，返回空字符串）

datetime 模块中不同对象的区别：
date 只表示日期。支持与 date 或 timedelta 进行加减操作.
time 只表示时分。不支持与 time 或 timedelta 进行加减操作，计算间隔需要先转换成 datetime 对象。
datetime 同时表示日期和时分的时间对象。同时具备 date 和 time 对象的行为和属性，可以从中解析出单独的 date 和 time 对象。
timedelta 表示两个时间之间的间隔。只通过 days 、seconds，microseconds 这 3 种单位来表示。

datatime模块重新封装了time模块，提供更多接口，提供的类有：date,time,datetime,timedelta,tzinfo。


1、date类

datetime.date(year, month, day)

静态方法和字段
date.max、date.min：date对象所能表示的最大、最小日期；
date.resolution：date对象表示日期的最小单位。这里是天。
date.today()：返回一个表示当前本地日期的date对象；
date.fromtimestamp(timestamp)：根据给定的时间戮，返回一个date对象；


方法和属性
d1 = date(2011,06,03)#date对象
d1.year、date.month、date.day：年、月、日；
d1.replace(year, month, day)：生成一个新的日期对象，用参数指定的年，月，日代替原有对象中的属性。（原有对象仍保持不变）
d1.timetuple()：返回日期对应的time.struct_time对象；
d1.weekday()：返回weekday，如果是星期一，返回0；如果是星期2，返回1，以此类推；
d1.isoweekday()：返回weekday，如果是星期一，返回1；如果是星期2，返回2，以此类推；
d1.isocalendar()：返回格式如(year，month，day)的元组；
d1.isoformat()：返回格式如'YYYY-MM-DD’的字符串；
d1.strftime(fmt)：和time模块format相同。


2、time类

datetime.time(hour[ , minute[ , second[ , microsecond[ , tzinfo] ] ] ] )

静态方法和字段
time.min、time.max：time类所能表示的最小、最大时间。其中，time.min = time(0, 0, 0, 0)， time.max = time(23, 59, 59, 999999)；
time.resolution：时间的最小单位，这里是1微秒；

方法和属性
t1 = datetime.time(10,23,15)#time对象
t1.hour、t1.minute、t1.second、t1.microsecond：时、分、秒、微秒；
t1.tzinfo：时区信息；
t1.replace([ hour[ , minute[ , second[ , microsecond[ , tzinfo] ] ] ] ] )：创建一个新的时间对象，用参数指定的时、分、秒、微秒代替原有对象中的属性（原有对象仍保持不变）；
t1.isoformat()：返回型如"HH:MM:SS"格式的字符串表示；
t1.strftime(fmt)：同time模块中的format；


3、datetime类

datetime相当于date和time结合起来。
datetime.datetime (year, month, day[ , hour[ , minute[ , second[ , microsecond[ , tzinfo] ] ] ] ] )

静态方法和字段
datetime.today()：返回一个表示当前本地时间的datetime对象；
datetime.now([tz])：返回一个表示当前本地时间的datetime对象，如果提供了参数tz，则获取tz参数所指时区的本地时间；
datetime.utcnow()：返回一个当前utc时间的datetime对象；#格林威治时间
datetime.fromtimestamp(timestamp[, tz])：根据时间戮创建一个datetime对象，参数tz指定时区信息；
datetime.utcfromtimestamp(timestamp)：根据时间戮创建一个datetime对象；
datetime.combine(date, time)：根据date和time，创建一个datetime对象；
datetime.strptime(date_string, format)：将格式字符串转换为datetime对象；


方法和属性
dt=datetime.now()#datetime对象
dt.year、month、day、hour、minute、second、microsecond、tzinfo：
dt.date()：获取date对象；
dt.time()：获取time对象；
dt.replace ([ year[ , month[ , day[ , hour[ , minute[ , second[ , microsecond[ , tzinfo] ] ] ] ] ] ] ])：
dt.timetuple ()
dt.utctimetuple ()
dt.toordinal ()
dt.weekday ()
dt.isocalendar ()
dt.isoformat ([ sep] )
dt.ctime ()：返回一个日期时间的C格式字符串，等效于time.ctime(time.mktime(dt.timetuple()))；
dt.strftime (format)

4.timedelta类，时间加减

使用timedelta可以很方便的在日期上做天days，小时hour，分钟，秒，毫秒，微妙的时间计算，如果要计算月份则需要另外的办法。



"""

#===================================================================================
#  datetime.date类
#===================================================================================
from datetime import  date


print("datetime.date类的类属性max：",date.max)
print("datetime.date类的类属性min：",date.min)
print("datetime.date类的类属性resoluation：",date.resolution)
print("datetime.date类的静态方法_today",date.today())
print("datetime.date类的静态方法_fromtimestamp",date.fromtimestamp(1117937981))
# datetime.date类的类属性max： 9999-12-31
# datetime.date类的类属性min： 0001-01-01
# datetime.date类的类属性resoluation： 1 day, 0:00:00
# datetime.date类的静态方法_today 2022-04-26
# datetime.date类的静态方法_fromtimestamp 2005-06-05



today= date.today()
print("today = {}".format(today))
print("当前日期为：",today)
# today = 2022-04-26
# 当前日期为： 2022-04-26


formatted_today=today.strftime('%y:%m:%d')
print("formatted_today = {}".format(formatted_today))
# formatted_today = 22:04:26


#实例对象的属性
print("当前时间的date对象为：",today)
print(type(today))#返回一个date对象
# 当前时间的date对象为： 2022-04-26
# <class 'datetime.date'>

print("实例对象date的year属性：",today.year)
print("实例对象date的month属性：",today.month)
print("实例对象date的day属性：",today.day)
# 实例对象date的year属性： 2022
# 实例对象date的month属性： 4
# 实例对象date的day属性： 26

print("实例对象date的isoformat()：",today.isoformat()) #返回格式如'YYYY-MM-DD'的字符串
print("实例对象date的timetuple() ：",today.timetuple() ) #返回时间元组struct_time格式的日期
print("实例对象date的toordinal()：",today.toordinal())#返回日期是是自 0001-01-01 开始的第多少天
print("实例对象date的weekday() ：",today.weekday() )#返回日期是星期几，[0, 6]，0表示星期一
print("实例对象date的isoweekday()：",today.isoweekday())#返回日期是星期几，[1, 7], 1表示星期一
print("实例对象date的isocalendar()：",today.isocalendar()) #返回一个元组，格式为：(year,week,weekday)。week：一年中的第几周，weekday：当前日期是星期几
# 实例对象date的isoformat()： 2022-04-26
# 实例对象date的timetuple() ： time.struct_time(tm_year=2022, tm_mon=4, tm_mday=26, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=116, tm_isdst=-1)
# 实例对象date的toordinal()： 738271
# 实例对象date的weekday() ： 1
# 实例对象date的isoweekday()： 2
# 实例对象date的isocalendar()： datetime.IsoCalendarDate(year=2022, week=17, weekday=2)


#实例对象的replace()方法 ,生成一个新的日期对象，用参数指定的年，月，日代替原有对象中的属性(原有对象仍保持不变)
new_day = today.replace(2019,1,1) #新生成一个date对象：年月日
new_day1 = today.replace(2019,1) #新生成一个date对象：年月
new_day2 = today.replace(month=2,day=2) #新生成一个date对象：月日
print("新生成的date对象为：",new_day)
print("新生成的date1对象为：",new_day1)
print("新生成的date2对象为：",new_day2)
print("旧的date对象不变：",today)
# 新生成的date对象为： 2019-01-01
# 新生成的date1对象为： 2019-01-26
# 新生成的date2对象为： 2022-02-02
# 旧的date对象不变： 2022-04-26

#实例对象的strftime()和isoformat()方法

from datetime import date

today = date.today()  #返回当前日期
print("当前日期为：",today)
# 当前日期为： 2022-04-26


new_day = today.strftime("%Y/%m/%d")    #重新格式化日期格式：返回自定义格式的时间字符串
new_day1 = today.strftime("%Y{0}%m{1}%d{2}").format("年","月","日")
print(new_day)
print(new_day1)
# 2022/04/26
# 2022年04月26日

new_day2 = today.isoformat()#再次重新格式化日期格式：这个方法是写死了格式的-
print(new_day2)
# 2022-04-26


#实例对象的timetuple()方法
from datetime import date
import time

today = date.today()    #返回当前日期
print("当前日期为：",today)
# 当前日期为： 2022-04-26


new_day = today.timetuple()
print(new_day)
#这里可以使用time类下面的方法将一个时间元组的值转化为字符串型的时间值
print(time.strftime("%Y-%m-%d %H:%M:%S", new_day))
# time.struct_time(tm_year=2022, tm_mon=4, tm_mday=26, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=116, tm_isdst=-1)
# 2022-04-26 00:00:00


#示例1：Python获取今天的日期
from datetime import date

today = date.today()
print("今天的日期:", today)
# 今天的日期: 2022-04-26


#示例2：当前日期以不同的格式
# dd/mm/YY
d1 = today.strftime("%d/%m/%Y")
print("d1 =", d1)
# d1 = 26/04/2022

# 文字的月、日、年
d2 = today.strftime("%B %d, %Y")
print("d2 =", d2)
# d2 = April 26, 2022


# mm/dd/y
d3 = today.strftime("%m/%d/%y")
print("d3 =", d3)
# d3 = 04/26/22

# 月份缩写，日期和年份
d4 = today.strftime("%b-%d-%Y")
print("d4 =", d4)
# d4 = Apr-26-2022

#===================================================================================
#  datetime.time类
#===================================================================================
#datetime模块下的time类由：hour小时、minute分钟、second秒、microsecond毫秒和tzinfo五部分组成


from datetime import time

Time = time(5)  #这里也是相当于实例化一个time
print("只有小时数：",Time)
# 只有小时数： 05:00:00


Time1 = time(hour=5,minute=12,second=12,microsecond=121212)
print("完整：",Time1)
print(type(Time1))  #返回一致time
# 完整： 05:12:12.121212
# <class 'datetime.time'>

tm = time(23, 46, 10)
print (  'tm:', tm)
print (  'hour: %d, minute: %d, second: %d, microsecond: %d' % (tm.hour, tm.minute, tm.second, tm.microsecond))
tm1 = tm.replace(hour=20)
print  ( 'tm1:', tm1)
print (  'isoformat():', tm.isoformat())
print (  'strftime()', tm.strftime("%X"))
# tm: 23:46:10
# hour: 23, minute: 46, second: 10, microsecond: 0
# tm1: 20:46:10
# isoformat(): 23:46:10
# strftime() 23:46:10


#datetime.time类的静态属性
from datetime import time

print(time.max)  # time类所能表示的最大时间：time(23, 59, 59, 999999)
print(time.min)  # time类所能表示的最小时间：time(0, 0, 0, 0)
print(time.resolution) # 时间的最小单位，即两个不同时间的最小差值：1微秒
# 23:59:59.999999
# 00:00:00
# 0:00:00.000001


from datetime import time

Time = time(22, 21, 45, 111)  #实例化一个time对象：Time
print("时间：",Time)

print("实例对象的hour属性：",Time.hour)
print("实例对象的minute属性：",Time.minute)
print("实例对象的second属性：",Time.second)
print("实例对象的microsecond属性：",Time.microsecond)
print("实例对象的tzinfo属性：",Time.tzinfo) #实例化时没有指明所用时区，因此为None
# 时间： 22:21:45.000111
# 实例对象的hour属性： 22
# 实例对象的minute属性： 21
# 实例对象的second属性： 45
# 实例对象的microsecond属性： 111
# 实例对象的tzinfo属性： None

from datetime import time

Time = time(22, 18, 45, 111)  #实例化一个time对象：Time
print("时间：",Time)
# 时间： 22:18:45.000111


#strftime()方法：根据传入的格式来输出时间字符串
print("指定格式：",Time.strftime("%H-%M-%S.%f"))
print("指定格式：",Time.strftime("%H{0}%M{1}%S{2}").format("时","分","秒")) #可以不格式化毫秒
# 指定格式： 22-18-45.000111
# 指定格式： 22时18分45秒

#isoformat()：输出固定格式的字符串
print("固定格式：",Time.isoformat())
# 固定格式： 22:18:45.000111

#replace()：生成新的time对象
print("生成新的time对象：",Time.replace(hour=3))
print("生成新的time对象：",Time.replace(3,4,5,43))
# 生成新的time对象： 03:18:45.000111
# 生成新的time对象： 03:04:05.000043



#===================================================================================
#  datetime.datetime类
# 、datetime模块下的datetime类是Python处理日期和时间的标准库。datetime类是date类与time类的结合体，包括date类与time类的所有信息
#===================================================================================


from datetime import datetime

now = datetime(2020,8,27,12,12,12,1111)#实例化一个datetime对象
print("当前日期为：",now)
print(type(now))
# 当前日期为： 2020-08-27 12:12:12.001111
# <class 'datetime.datetime'>

now1 = datetime(year=2020,month=8,day=27) #年月日为必填参数，其余未传入时默认为0
print("当前日期为：",now1)
# 当前日期为： 2020-08-27 00:00:00

"""
datetime.datetime类的静态属性
1、以下是datetime.datetime类的一些(类)静态方法和属性：直接通过类名调用

类属性、类方法	描述
datetime.today() 	返回一个表示当前本地时间的datetime对象
datetime.now([tz]) 	返回一个表示当前本地时间的datetime对象，如果提供了参数tz，则获取tz参数所指时区的本地时间
datetime.utcnow()	返回一个当前utc时间的datetime对象；#格林威治时间
datetime.fromtimestamp(timestamp[, tz]) 	根据时间戮创建一个datetime对象，参数tz指定时区信息
datetime.utcfromtimestamp(timestamp)	根据时间戮创建一个datetime对象(utc)
datetime.combine(date, time)  	根据date和time，创建一个datetime对象
datetime.strptime(date_string, format)	将格式字符串转换为datetime对象

"""
from datetime import datetime
import time
print  ( 'datetime.max:', datetime.max)
print  ( 'datetime.min:', datetime.min)
print  ( 'datetime.resolution:', datetime.resolution)
print  ( 'today():', datetime.today())
print  ( 'now():', datetime.now())
print  ( 'utcnow():', datetime.utcnow())
print  ( 'fromtimestamp(time.time()):', datetime.fromtimestamp(time.time()))
print  ( 'utcfromtimestamp(time.time()):', datetime.utcfromtimestamp(time.time()))
# datetime.max: 9999-12-31 23:59:59.999999
# datetime.min: 0001-01-01 00:00:00
# datetime.resolution: 0:00:00.000001
# today(): 2022-04-26 22:42:34.170496
# now(): 2022-04-26 22:42:34.171119
# utcnow(): 2022-04-26 14:42:34.171177
# fromtimestamp(time.time()): 2022-04-26 22:42:34.171236
# utcfromtimestamp(time.time()): 2022-04-26 14:42:34.171298

"""
today()
1、today()方法的语法如下：datetime.datetime.today()

2、此语法中datetime.datetime指的是datetime模块中的datetime类

3、返回值：返回一个表示当前本地时间的datetime对象
"""
import datetime

now = datetime.datetime.today()
print("当前时间为：",now)
# 当前时间为： 2022-04-26 22:43:19.755650

print("当前时间为：",datetime.datetime.today())
# 当前时间为： 2022-04-26 22:43:19.755879


from datetime import datetime
print("当前时间为：",datetime.today())
# 当前时间为： 2022-04-26 22:43:37.813263

now = datetime.today()
print("当前时间为：",now)
# 当前时间为： 2022-04-26 22:43:48.349421

import datetime
today=  datetime.datetime.today()
print("today = {}".format(today))
# today = 2022-04-26 22:44:00.188934

formatted_today=today.strftime('%Y-%m-%d,%H:%M:%S')
print("formatted_today = {}".format(formatted_today))
# formatted_today = 2022-04-26,22:44:00


today=  datetime.today()
print("today = {}".format(today))
# today = 2022-04-26 22:44:51.452773

formatted_today=today.strftime('%Y-%m-%d,%H:%M:%S')
print("formatted_today = {}".format(formatted_today))
# formatted_today = 2022-04-26,22:44:51



"""
now()
1、now()方法的语法如下：datetime.datetime.now([tz])

2、此语法中datetime.datetime指的是datetime模块中的datetime类

3、如果提供参数tz，就获取tz参数所指时区的本地时间(now是本地时间，可以认为是你电脑现在的时间)

4、返回值：返回当前指定的本地时间，一个datetime对象

"""
from datetime import datetime
print("当前时间为：",datetime.now())
# 当前时间为： 2022-04-26 22:45:25.074636


#示例3：获取当前日期和时间
from datetime import datetime

# 包含当前日期和时间的datetime对象
now = datetime.now()
print("now =", now)
# now = 2022-04-26 22:45:33.989925


# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
# date and time = 26/04/2022 22:45:33
#

import datetime
# 获取年月日
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
print("current_date = {}".format(current_date))
# current_date = 2022-04-26

# 获取年月
current_date = datetime.datetime.now().strftime('%Y-%m')
print("current_date = {}".format(current_date))
# current_date = 2022-04

current_date = datetime.datetime.now().date()
print("current_date = {}".format(current_date))
# current_date = 2022-04-26

# 获取年份
current_date = datetime.datetime.now().strftime('%Y')
print("current_date = {}".format(current_date))
# current_date = 2022


from datetime import datetime
import time
# 获取当前时间
now_time =  datetime.now()
print(f"now_time = {now_time}")
print(f"datetime.now() = {datetime.now()}")
# now_time = 2022-04-26 23:02:35.489473
# datetime.now() = 2022-04-26 23:02:35.489599


# 格式化时间字符串
str_time = now_time.strftime("%Y-%m-%d %X")
print("str_time = {}".format(str_time))
#  str_time = 2022-04-26 22:47:00

tup_time = time.strptime(str_time,"%Y-%m-%d %X")
print("tup_time = {}".format(tup_time))
print("year = {}, mon = {}, day = {}, hour = {}, min = {}. sec = {} ".format(tup_time.tm_year,tup_time.tm_mon,tup_time.tm_mday,tup_time.tm_hour,tup_time.tm_min,tup_time.tm_sec))
# tup_time = time.struct_time(tm_year=2022, tm_mon=4, tm_mday=26, tm_hour=23, tm_min=0, tm_sec=51, tm_wday=1, tm_yday=116, tm_isdst=-1)
# year = 2022, mon = 4, day = 26, hour = 23, min = 0. sec = 51 


time_sec = time.mktime(tup_time)
print("time_sec = {}".format(time_sec))
# time_sec = 1650985251.0


# 转换成时间戳 进行计算
time_sec += 1
tup_time2 = time.localtime(time_sec)
str_time2 = time.strftime("%Y-%m-%d %X",tup_time2)

print("str_time2 = {}".format(str_time2))
# str_time2 = 2022-04-26 23:00:52





"""
utcnow()
1、utcnow()方法的语法如下：datetime.datetime.utcnow()

2、此语法中datetime.datetime指的是datetime模块中的datetime类

3、返回值：返回一个当前utc时间的datetime对象
"""
from datetime import datetime

print("当前时间为：",datetime.utcnow())
print("当前时间为：",datetime.now())
# 当前时间为： 2022-04-26 15:09:10.671105
# 当前时间为： 2022-04-26 23:09:10.671231

"""
fromtimestamp()
1、fromtimestamp()方法的语法如下：datetime.datetime.fromtimestamp(timestamp[,tz]

2、此语法中datetime.datetime指的是datetime模块中的datetime类

3、参数tz表示指定的时区信息，参数timestamp表示需要转换的时间戳

4、返回值：将一个时间戳形式的时间转换为可读形式，返回一个datetime对象

"""

from datetime import datetime
import time

now = time.time()
print("当前时间戳：",now)
# 当前时间戳： 1650985781.004065

Time = datetime.fromtimestamp(now)
print("当前时间为：",Time)
# 当前时间为： 2022-04-26 23:09:41.004065

print("当前时间为：",datetime.fromtimestamp(1583355285))
# 当前时间为： 2020-03-05 04:54:45

"""
combine()
1、combine()方法的语法如下：datetime.datetime.combine(date, time)

2、此语法中datetime.datetime指的是datetime模块中的datetime类

3、参数date表示一个date对象，参数time表示一个time对象

4、返回值：根据date和time，创建一个datetime对象
"""


import datetime

Date = datetime.date.today()
print("当前日期为：",Date)

Time = datetime.time(12,12,12,122)
print("当前时间为：",Time)

now = datetime.datetime.combine(Date,Time)
print("当前日期时间为：",now)

# 当前日期为： 2022-04-26
# 当前时间为： 12:12:12.000122
# 当前日期时间为： 2022-04-26 12:12:12.000122



"""
strptime()
1、作用：将时间型字符串格式化为datetime对象

2、strptime()方法的语法如下：datetime.datetime.strptime(date_string,format)

3、此语法中datetime.datetime指的是datetime模块中的datetime类

4、参数date_string指日期字符串，format为格式方式

5、返回值：返回一个datetime对象
"""

from datetime import datetime

"""
1、strptime()方法将一个字符串时间格式化为一个datetime对象
2、目标格式中传入了.%f属性时，传入的时间字符串中必须含有毫秒(必须为对应关系)
3、目标格式中未传入了.%f属性时，传入的时间字符串中不能含有毫秒(必须为对应关系)
"""
time1 = datetime.strptime("2018-08-19 19:23:57.1111", "%Y-%m-%d %H:%M:%S.%f")
print(time1)
print(type(time1))
# 2018-08-19 19: 23: 57.111100
# <class 'datetime.datetime' >

time2 = datetime.strptime("2018-08-19 19:23:57", '%Y-%m-%d %H:%M:%S')
print(time2)
# 2018-08-19 19: 23: 57


from datetime import datetime

"""
将一个字符串时间值格式化成一个datetime对象时
    传入的datetime格式必须与字符串时间值格式一致，否则会报错
    有无毫秒、时间值连接符都必须一致
"""

#time3 = datetime.strptime("2021-08-24 13:33:11.222","%Y|%m-%d %H-%M:%S.%f")
#print(time3)



import datetime

input_date = str(input("-----："))

try:
    date = datetime.datetime.strptime(input_date, '%Y-%m-%d %H:%M:%S')
    print("当前输入的时间值为：", date)
except:
    print("传入的时间值错误")

print("时间值校验完成")
# -----：2020-04-26 23:12:12
# 当前输入的时间值为： 2020-04-26 23:12:12
# 时间值校验完成

"""
datetime类对象的方法和属性
1、实例化一个类对象(datetime类)或返回一个datetime类对象后，通过其实例名来调用

2、以下方法和属性需要通过：datetime.datetime类的实例对象调用

3、datetime.datetime类的类方法或属性返回的如果是一个datetime对象，那这个datetime对象又有自己的方法或属性
    ⑴一般而言，在Python中返回值是一个对象，那么这个对象都有自己的方法或属性

对象方法	                                  描述(dt表示datetime类的实例对象)
dt.year, dt.month, dt.day                    	年、月、日
dt.hour, dt.minute, dt.second	                 时、分、秒
dt.microsecond, dt.tzinfo	                  微秒、时区信息
dt.date()	                              获取datetime对象对应的date对象
dt.time()	                       获取datetime对象对应的time对象， tzinfo 为None
dt.timetuple()	                            返回datetime对象对应的tuple（不包括tzinfo）
dt.timetz()	                            获取datetime对象对应的time对象，tzinfo与datetime对象的tzinfo相同
dt.utctimetuple()	                  返回datetime对象对应的utc时间的tuple（不包括tzinfo）
dt.toordinal()	                                同date对象
dt.weekday()	                           同date对象
dt.isocalendar()	                  同date对象
dt.isoformat([sep])               	返回一个‘%Y-%m-%d
dt.ctime()    	                  等价于time模块的time.ctime(time.mktime(d.timetuple()))
dt.strftime(format)	                  返回指定格式的时间字符串
dt.replace(year, month, day, hour,minute,second, microsecond, tzinfo)	生成并返回一个新的datetime对象
strftime()
1、作用：将datetime对象格式化为字符串

2、strftime()方法的语法如下：dt.strftime(format)

3、此语法中datetime.datetime指的是datetime模块中的datetime类

4、dt表示datetime类的实例对象，参数format为格式化方式

3、返回值：返回一个时间字符串

"""



from datetime import datetime

"""
1、strftime方法将一个datetime对象格式化为一个字符串时间
2、目标格式中传入了.%f属性时，输出的时间值中将会有毫秒
3、目标格式中未传入了.%f属性时，输出的时间值中将不会有毫秒
"""
now = datetime.now()
print("当前时间",now)
print(type(now))
# 当前时间 2022-04-26 23:12:39.819306
# <class 'datetime.datetime'>

time1 = now.strftime("%Y-%m-%d %H:%M:%S.%f")
print(time1)
# 2022-04-26 23:12:39.819306

time2 = now.strftime('%Y-%m-%d %H:%M:%S')
print(time2)
print(type(time1))
# 2022-04-26 23:12:39
# <class 'str'>



from datetime import datetime

now = datetime.now()
print(now)
# 2022-04-26 23:13:31.673095

#格式化的字符串时间值带6位毫秒
stringTime = now.strftime("%Y|%m|%d| %H-%M-%S.%f")
print(stringTime)
# 2022|04|26| 23-13-31.673095

#格式化的字符串时间值带3位毫秒
stringTime1 = now.strftime("%Y|%m|%d| %H-%M-%S.%f")[:-3]
print(stringTime1)
# 2022|04|26| 23-13-31.673

#格式化的字符串时间值不带毫秒
stringTime2 = now.strftime("%Y|%m|%d| %H-%M-%S")
print(stringTime2)
# 2022|04|26| 23-13-31

#将不带毫秒的datetime对象转为带毫秒的时间字符串
time3 = datetime.strptime("2021-08-24 13:33:11","%Y-%m-%d %H:%M:%S")
stringTime3 = time3.strftime("%Y-%m-%d %H:%M:%S.%f")
print(stringTime3)
# 2021-08-24 13:33:11.000000

"""

1、将一个datetime对象转为字符串时间值时：
    不需要考虑datetime对象与字符串时间值格式之间的关系：两个不需要一样
    有无毫秒、时间值连接符都可以不对应
2、%Y、%m、%d、%H、%M、%S、%f对应年月日时分秒毫秒：它们是有实际格式化意义的，代表具体的值，在格式化时会用具体的值去填充
    而前面用到的"-"、":"、"|"这些是没有实际格式化意义的，因此直接输出
    就跟字符串格式化一样：%s只代表某个具体的值，其他无格式化意义的，就直接输出

"""



import datetime,time

today = datetime.datetime.now()

print(f"today.year = {today.year}")
print(f"today.hour = {today.hour}")
print(f"today.month = {today.month}")
print(f"today.minute = {today.minute}")
print(f"today.second = {today.second}")
print(f"today.microsecond = {today.microsecond}")
print(f"today.tzinfo = {today.tzinfo}")
print(f"today.date() = {today.date()}")
print(f"today.time() = {today.time()}")
# today.year = 2022
# today.hour = 23
# today.month = 4
# today.minute = 17
# today.second = 23
# today.microsecond = 842433
# today.tzinfo = None
# today.date() = 2022-04-26
# today.time() = 23:17:23.842433







#===================================================================================
#  datetime.timedelta类
#===================================================================================
"""

datetime.timedelta类
1、timedelta对象表示两个不同时间之间的差值
    ⑴datetime.timedelta对象代表两个时间之间的时间差，两个date、time或datetime对象相减就可以返回一个timedelta对象

2、如果使用time模块对时间进行算术运行，只能将字符串格式的时间和struct_time格式的时间对象先转换为时间戳格式
    ⑴然后对该时间戳加上或减去n秒，最后再转换回struct_time格式或字符串格式，这显然很不方便

3、而datetime模块提供的timedelta类可以让我们很方面的对datetime.date, datetime.time和datetime.datetime对象做算术运算
    ⑴且两个时间之间的差值单位也更加容易控制。这个差值的单位可以是：天、秒、微秒、毫秒、分钟、小时、周


构造函数
datetime.timedelta类的构造函数如下：

datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
#天、秒、微秒、毫秒、分钟、小时、周
注：
1、所有参数都是默认参数，默认值为0

2、参数的值可以是整数或浮点数，也可以是正数或负数(正数表示加，负数表示减)

3、针对时间存储，timedelta内部只能存储days，seconds，microseconds，其他参数的值会自动按如下规则进行转换：
    ⑴millisecond毫秒）转换成1000 microseconds（微秒）
    ⑵minute转换成60 seconds
    ⑶hour转换成3600 seconds
    ⑷week转换成7 days

"""




from datetime import timedelta

# 实例化一个时间段值：1周2天3小时4分钟5秒777毫秒666微秒
time1 = timedelta(days=2, seconds=5, microseconds=666, milliseconds=777, minutes=4, hours=3, weeks=1)
print(time1)
print(type(time1))

time2 = timedelta(days=2, hours=1)
print(time2)
# 9 days, 3:04:05.777666
# <class 'datetime.timedelta'>
# 2 days, 1:00:00


import datetime

Date = datetime.datetime.today()
print("当前日期：",Date)
# 当前日期： 2022-04-26 23:18:34.638084

Date1 = datetime.datetime(2020,8,20,12,12,12)
print("过去日期为：",Date1)
# 过去日期为： 2020-08-20 12:12:12

d = Date - Date1  #这里以datetime对象为例，date对象与time对象也是可以相减的
print("日期差为：",d)
print(type(d))
# 日期差为： 614 days, 11:06:22.638084
# <class 'datetime.timedelta'>


import datetime

#print(datetime.timedelta(365).total_seconds())  #一年中共有多少秒

today = datetime.datetime.now()
print("当前时间：",today)
# 当前时间： 2022-04-26 23:19:26.790898

three_day_later = today + datetime.timedelta(3)   #3天后
print(three_day_later)
print(type(three_day_later)) #还是一个datetime对象
# 2022-04-29 23:19:26.790898
# <class 'datetime.datetime'>

three_day_befor = today - datetime.timedelta(days = 3)   #3天前
#three_day_befor = today + datetime.timedelta(-3)
print(three_day_befor)
# 2022-04-23 23:19:26.790898

three_hours_later = today + datetime.timedelta(hours=3)   #3小时后
print(three_hours_later)
# 2022-04-27 02:19:26.790898

new_day = today + datetime.timedelta(days=1,minutes=22,seconds=30,milliseconds=2)   #1天22分钟30秒2毫秒后
print(new_day)
# 2022-04-27 23:41:56.792898

"""
datetime.timedelta类对象的方法和属性
1、实例化一个类对象后(timedelta类)或返回一个timedelta类对象后，通过其实例名来调用

2、以下方法和属性需要通过：datetime.timedelta类的实例对象调用

实例方法/属性名称	 描述(td表示timedelta类的实例对象)
td.days 	天 [-999999999, 999999999]：返回时间差的天部分
td.seconds	 秒 [0, 86399]：返回之间差的时分秒部分(转换为秒返回)
td.microseconds	微秒 [0, 999999]
td.total_seconds() 	时间差中包含的总秒数
datetime.datetime.now()  	返回当前本地时间（datetime.datetime对象实例）
datetime.datetime.fromtimestamp(timestamp)  	返回指定时间戳对应的时间（datetime.datetime对象实例）
datetime.timedelta()    	返回一个时间间隔对象，可以直接与datetime.datetime对象做加减操作


"""


from datetime import datetime, timedelta

future = datetime.strptime("2020-02-01 08:00:00", "%Y-%m-%d %H:%M:%S")
now = datetime.now()

TimeEquation = now - future
print(TimeEquation)            #返回两个时间差的天、时分秒部分：49 days, 5:58:56.887765
print(TimeEquation.days)    #只返回两个时间差的天数部分：49
print(TimeEquation.seconds)    #只返回两个时间差的时分秒部分(换算为秒返回)：21536=(5*60*60)+(58*60)+56
print(TimeEquation.seconds/60/60)#只返回两个时间差的时分秒部分(换算为小时返回)：5.982222222222222=5+(59/60)
print(TimeEquation.total_seconds())
# 815 days, 15:20:08.256429
# 815
# 55208
# 15.335555555555555
# 70471208.256429

#返回两个时间差的总秒数(将相差天、时、分、秒都转为秒)：4255136.887765=(48*24*60*60)+(5*60*60)+(58*60)+56

"""
补充
计算两个时间的时间差
其本上常用的类有：datetime和timedelta两个。它们之间可以相互加减。每个类都有一些方法和属性可以查看具体的值
    ⑴ datetime可以查看：天数(day)，小时数(hour)，星期几(weekday())等
    ⑵ timedelta可以查看：天数(days)，秒数(seconds)等

"""


from datetime import datetime,timedelta

future = datetime.strptime("2032-02-01 08:00:00","%Y-%m-%d %H:%M:%S")
now = datetime.now()

TimeEquation = future - now      #计算时间差
#print(TimeEquation)              #145 days, 20:50:32.599774

hours = TimeEquation.seconds/60/60
minutes = (TimeEquation.seconds - hours*60*60)/60
seconds = TimeEquation.seconds - hours*60*60 - minutes * 60

print("今天是：",now.strftime("%Y-%m-%d %H:%M:%S"))
print("距离2032-02-01，还剩下%d天" % TimeEquation.days)
print(TimeEquation.days, hours, minutes, seconds)
# 今天是： 2022-04-26 23:20:23
# 距离2019-02-01，还剩下3567天
# 3567 8.66 0.0 0.0





#===================================================================================
#  Python中的时区转换
#===================================================================================

"""
Python中的时区转换
1、在Python中将一个时区的时间转换成另一个时区的时间，比较常用的方法是：结合使用datetime模块和pytz模块，一般步骤为
    ⑴使用datetime模块来获取一个时间对象(datetime对象)
    ⑵使用pytz模块来创建一个将要转换到的时区对象
    ⑶使用datetime对象方法astimezone()方法来进行转换

2、在介绍时区转换前需要介绍两个时间概念：
    ⑴navie时间：获取的时间对象不知道自己的时间表示的是哪个时区的(如：使用datetime.now()方法获取的时间对象就不知道当前表示的是哪个时区)
    ⑵aware时间：获取的时间对象知道自己的时间表示的是哪个时区的

"""

from datetime import datetime,timedelta,timezone
import pytz

now = datetime.now()
print(now)

utc_timezone = pytz.timezone("UTC")#使用pytz.timezone()方法来创建一个时区对象
utc_now = now.astimezone(utc_timezone)#使用astimezone()方法来将时间转换为另一个时区的时间



"""
注：
1、从上面例子的输出可以看出，在进行时区转换时(now.astimezone(utc_timezone))，报错了，这是因为：
    ⑴在使用datetime.now()获取的时间对象是一个navie时间，即不知道自己的时间表示的是哪个时区。既然不知道自己的时间当前在哪个时区，那肯定是不能将当前时间转化为另一个时区的时间的
    ⑵要解决这个问题，因此需要将使用datetime.now()获取的这个时间加上一个时区标志
    ⑶要给一个无时区标志的时间加上一个时区标志，那就可以用到datetime对象下的replace()方法了

2、pytz.timezone(‘时区名’)：此方法能获取一个tzinfo对象，该对象可在datetime生成时间中以参数的形式放入，即可生成对应时区的时间

3、astimezone()：datetime对象方法，用于将一个时区的时间转为另一个时区的时间。这个方法只能别aware类型时间调用，不能被navie类型的时间调用

4、replace()：datetime对象方法，可以将一个时间的某些属性进行修改

5、因此正确的将一个时区的时间转为另一个时区的时间，可以使用下面这种方法

"""






from datetime import datetime,timedelta,timezone
import pytz

now = datetime.now()
print(now)

utc_timezone = pytz.timezone("UTC")

now = now.replace(tzinfo=pytz.timezone("Asia/Shanghai"))
print(now)

utc_now = now.astimezone(utc_timezone)
print(utc_now)


from datetime import datetime, tzinfo,timedelta
"""
tzinfo是关于时区信息的类
tzinfo是一个抽象类，所以不能直接被实例化
"""
class UTC(tzinfo):
    """UTC"""
    def __init__(self,offset = 0):
        self._offset = offset

    def utcoffset(self, dt):
        return timedelta(hours=self._offset)

    def tzname(self, dt):
        return "UTC +%s" % self._offset

    def dst(self, dt):
        return timedelta(hours=self._offset)

beijing = datetime(2011,11,11,0,0,0,tzinfo = UTC(8))
print( "beijing time:",beijing)
#曼谷时间
bangkok = datetime(2011,11,11,0,0,0,tzinfo = UTC(7))
print ("bangkok time",bangkok)
#北京时间转成曼谷时间
print ("beijing-time to bangkok-time:",beijing.astimezone(UTC(7)))

#计算时间差时也会考虑时区的问题
timespan = beijing - bangkok
print ("时差:",timespan)




"""
日历模块
1、此模块的函数都是日历相关的，例如打印某月的字符月历

2、星期一是默认的每周第一天，星期天是默认的最后一天。更改设置需调用calendar.setfirstweekday()函数

calendar()
1、语法：calendar.calendar(year,w=2,l=1,c=6)

2、作用：返回一个多行字符串格式的year年年历，3个月一行，间隔距离为c。 每日宽度间隔为w字符。每行长度为21* W+18+2* C。l是每星期行数


"""





import calendar

print(calendar.calendar(1993))



"""
month()
1、语法：calendar.month(year,month,w=2,l=1)

2、返回一个多行字符串格式的year年month月日历，两行标题，一周一行。每日宽度间隔为w字符。每行的长度为7* w+6。l是每星期的行数
"""




import calendar

print(calendar.month(1993,7))



"""
isleap()
1、语法：calendar.isleap(year)

2、返回值：是闰年返回True，否则为false
"""



import calendar

month = calendar.isleap(1993)

if month == False:
    print("1993年不是闰年")

#上面代码的输出结果为：1993年不是闰年




"""
leapdays()
1、语法：calendar.leapdays(y1,y2)

2、返回值：返回在Y1，Y2两年之间的闰年总数
"""

import calendar

print(calendar.leapdays(2008,2018))

#上面代码的输出结果为：3



import datetime
# 获取当天最小时间
print("当天最小时间 = {}".format(datetime.datetime.combine(datetime.date.today(), datetime.time.min)))
# 当天最小时间 = 2022-04-26 00:00:00

# 获取当天最大时间
print("获取当天最大时间 = {}".format(datetime.datetime.combine(datetime.date.today(), datetime.time.max)))
# 获取当天最大时间 = 2022-04-26 23:59:59.999999

# 获取明天
print("明天 = {}".format(datetime.date.today() + datetime.timedelta(days=1)))
# 明天 = 2022-04-27

# 获取zuotian
print("zuo天 = {}".format(datetime.date.today() - datetime.timedelta(days=1)))
# zuo天 = 2022-04-25


#获取本周或本月第一天及最后一天
d_today = datetime.date.today()
print("zuo天 = {}".format(datetime.date.today()))
# zuo天 = 2022-04-26

# 获取本周第一天
FirstdayOfThisWeek = d_today - datetime.timedelta(d_today.weekday())
print("获取本周第一天 = {}".format(FirstdayOfThisWeek))
# 获取本周第一天 = 2022-04-25

# 获取本周最后一天
LastdayOfThisWeek = d_today + datetime.timedelta(6-d_today.weekday())
print("获取本周最后一天 = {}".format(LastdayOfThisWeek))
# 获取本周最后一天 = 2022-05-01




#按天生成日志
import os
import datetime,time



def createFile(filePath):
    """
    #参数为路径，先判断路径文件是否存在，然后尝试直接新建文件夹，如果失败，说明路径是多层路径(还可能有别的原因，这里一般情况够用了)，所以用makedirs创建多层文件夹。
    """
    if os.path.exists(filePath):
        print('%s:存在'%filePath)
    else:
        try:
            os.mkdir(filePath)
            print('新建文件夹：%s'%filePath)
        except Exception as e:
            os.makedirs(filePath)
            print('新建多层文件夹：%s' % filePath)


def make_dir(dir_name):
    make_file_path = os.path.join(dir_name+"{}".format(str(datetime.datetime.now().year)
    + "_" + str(datetime.datetime.now().month) + "_" + str(datetime.datetime.now().day)))
    make_file_name = make_file_path + "_log.txt"
    print(make_file_name)
    return make_file_name


def open_file(log_file_path):
    if not os.path.exists(log_file_path):
        os.system(r"touch {}".format(log_file_path))#调用系统命令行来创建文件
    log_file = os.open(log_file_path,"w")
    print(" = {}".format(log_file.name))
    return log_file


if __name__ == "__main__":
    dirname = '/home/jack/tmp/log/'
    createFile(dirname)
    make_file_name = make_dir(dirname)

    log_file = open_file(make_file_name)
    log_file.write("hello,jack")
    log_file.close()


