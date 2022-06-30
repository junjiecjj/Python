#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:09:08 2022

@author: jack

显示方式: 0（默认值）、1（高亮，即加粗）、4（下划线）、7（反显）、
前景色: 30（黑色）、31（红色）、32（绿色）、 33（黄色）、34（蓝色）、35（梅色）、36（青色）、37（白色）
背景色: 40（黑色）、41（红色）、42（绿色）、 43（黄色）、44（蓝色）、45（梅色）、46（青色）、47（白色）

 # 显示格式: \033[显示方式;前景色;背景色m
# ------------------------------------------------
# 显示方式             说明
#   0                 终端默认设置
#   1                 高亮显示
#   4                 使用下划线
#   5                 闪烁
#   7                 反白显示
#   8                 不可见
#   22                非粗体
#   24                非下划线
#   25                非闪烁
#
#   前景色             背景色            颜色
#     30                40              黑色
#     31                41              红色
#     32                42              绿色
#     33                43              黃色
#     34                44              蓝色
#     35                45              紫红色
#     36                46              青蓝色
#     37                47              白色

四 常见开头格式：
\033[0m      默认字体正常显示，不高亮

\033[32;0m      红色字体正常显示  

\033[1;32;40m  显示方式: 高亮    字体前景色：绿色  背景色：黑色

\033[0;31;46m  显示方式: 正常    字体前景色：红色  背景色：青色 

"""
print("\033[5;31;46m  test..1.................    \033[0m")

print("\033[5;31;40mLinux公社www.linuxidc.com\033[0m") #闪烁；红色；黑色背景 Linux公社www.linuxidc.com


print("\033[4;32;47m Linux公社www.linuxidc.com\033[0m") #下划线；绿色；白色背景 Linux公社www.linuxidc.com

print("\033[1;37;40mLinux公社www.linuxidc.com\033[0m") #高亮；白色；黑色背景 Linux公社www.linuxidc.com


print("\033[1;31;40m您输入的帐号或密码错误！\033[0m") 

print("\033[1;31;40m您输入的帐号或密码错误！")      




print("\033[0;31m %s \033[0m" % '输出红色字符')#  标准写法


print("\033[31m %s \033[0m" % '输出红色字符')#　 显示方式为0时，可以省略

print("\033[31m %s" % "输出红色字符")

print("\033[31m a = %d, f = %f, s = %s  \033[0m"%(12,12.13,'sasada'))

print("\033[5;31;46m  test..2.................    \033[0m")

class bcolors(object):
    OK = '\033[32m' #GREEN
    WARNING = '\033[33m' #YELLOW
    FAIL = '\033[31m' #RED
    RESET = '\033[0m' #RESET COLOR

print(bcolors.OK + "File Saved Successfully!" + bcolors.RESET)
print(bcolors.WARNING + "Warning: Are you sure you want to continue?" + bcolors.RESET)
print(bcolors.FAIL + "Unable to delete record." + bcolors.RESET)

print(f"{bcolors.OK}File Saved Successfully %d %f %s!{bcolors.RESET}"%(21,32.45,'hello???'))
print(f"{bcolors.WARNING}Warning: Are you sure you want to continue?{bcolors.RESET}")
print(f"{bcolors.FAIL}Unable to delete record.{bcolors.RESET}")
#print(f"{bcolors.FAIL}Unable to delete record {} {} {} .{bcolors.RESET}".format(54,657.87,'jackkkk'))


print("\033[5;31;46m  test..3.................    \033[0m")

class Colored1(object):
    # 显示格式: \033[显示方式;前景色;背景色m
    # 只写一个字段表示前景色,背景色默认
    RED = '\033[31m'       # 红色
    GREEN = '\033[32m'     # 绿色
    YELLOW = '\033[33m'    # 黄色
    BLUE = '\033[34m'      # 蓝色
    FUCHSIA = '\033[35m'   # 紫红色
    CYAN = '\033[36m'      # 青蓝色
    WHITE = '\033[37m'     # 白色

    #: no color
    RESET = '\033[0m'      # 终端默认颜色

    def color_str(self, color, s):
        return '{}{}{}'.format(
            getattr(self, color),
            s,
            self.RESET
        )

    def red(self, s):
        return self.color_str('RED', s)

    def green(self, s):
        return self.color_str('GREEN', s)

    def yellow(self, s):
        return self.color_str('YELLOW', s)

    def blue(self, s):
        return self.color_str('BLUE', s)

    def fuchsia(self, s):
        return self.color_str('FUCHSIA', s)

    def cyan(self, s):
        return self.color_str('CYAN', s)

    def white(self, s):
        return self.color_str('WHITE', s)

# ----------使用示例如下:-------------
color = Colored1()
print (color.red('I am red! %d %f %s '%(12,13.43,'jack')))
print (color.green('I am gree!{} {} {} '.format(12,13.43,'jack')))
print (color.yellow('I am yellow!'))
print (color.blue('I am blue!'))
print (color.fuchsia('I am fuchsia!'))
print (color.cyan('I am cyan!'))
print (color.white('I am white'))


"""
# -----------------colorama模块的一些常量---------------------------
# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL
#
"""
print("\033[5;31;46m  test..4.................    \033[0m")

import colorama
from colorama import Fore
from colorama import Style

colorama.init()
print(Fore.BLUE + Style.BRIGHT + "This is the color of the sky" + Style.RESET_ALL)
print(Fore.GREEN + "This is the color of grass" + Style.RESET_ALL)
print(Fore.BLUE + Style.DIM + "This is a dimmer version of the sky" + Style.RESET_ALL)
print(Fore.YELLOW + "This is the color of the sun" + Style.RESET_ALL)


print("\033[5;31;46m  test..5.................    \033[0m")

from colorama import  init, Fore, Back, Style
init(autoreset=True)
class Colored(object):

    #  前景色:红色  背景色:默认
    def red(self, s):
        return Fore.RED + s + Fore.RESET

    #  前景色:绿色  背景色:默认
    def green(self, s):
        return Fore.GREEN + s + Fore.RESET

    #  前景色:黄色  背景色:默认
    def yellow(self, s):
        return Fore.YELLOW + s + Fore.RESET

    #  前景色:蓝色  背景色:默认
    def blue(self, s):
        return Fore.BLUE + s + Fore.RESET

    #  前景色:洋红色  背景色:默认
    def magenta(self, s):
        return Fore.MAGENTA + s + Fore.RESET

    #  前景色:青色  背景色:默认
    def cyan(self, s):
        return Fore.CYAN + s + Fore.RESET

    #  前景色:白色  背景色:默认
    def white(self, s):
        return Fore.WHITE + s + Fore.RESET

    #  前景色:黑色  背景色:默认
    def black(self, s):
        return Fore.BLACK

    #  前景色:白色  背景色:绿色
    def white_green(self, s):
        return Fore.WHITE + Back.GREEN + s + Fore.RESET + Back.RESET

color = Colored()
print (color.red('I am red! %d %f %s '%(12,13.43,'jack')))
print(color.green('I am gree!{} {} {} '.format(12,13.43,'jack')))
print (color.yellow('I am yellow!'))
print (color.blue('I am blue!'))
print (color.magenta('I am magenta!'))
print (color.cyan('I am cyan!'))
print (color.white('I am white!'))
print (color.white_green('I am white green!'))


