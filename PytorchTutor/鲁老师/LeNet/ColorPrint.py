#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:41:36 2022

@author: jack
"""


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

"""

class ColoPrint(object):
    # 显示格式: \033[显示方式;前景色;背景色m
    # 只写一个字段表示前景色,背景色默认
    BLACK = '\033[30m'       # 红色
    RED = '\033[31m'       # 红色
    GREEN = '\033[32m'     # 绿色
    YELLOW = '\033[33m'    # 黄色
    BLUE = '\033[34m'      # 蓝色
    FUCHSIA = '\033[35m'   # 紫红色
    CYAN = '\033[36m'      # 青蓝色
    WHITE = '\033[37m'     # 白色

    HigBLACK = '\033[1;30m'       # 
    HigRED = '\033[1;31m'       # 红色
    HigGREEN = '\033[1;32m'     # 绿色
    HigYELLOW = '\033[1;33m'    # 黄色
    HigBLUE = '\033[1;34m'      # 蓝色
    HigFUCHSIA = '\033[1;35m'   # 紫红色
    HigCYAN = '\033[1;36m'      # 青蓝色
    HigWHITE = '\033[1;37m'     # 白色
    
    HigBlackFG_WhiteBG = '\033[1;30;40m'       # 
    HigRedFG_WhiteBG = '\033[1;31;40m'       # 红色
    HigGreenFGWhiteBG = '\033[1;32;40m'     # 绿色
    HigYellowFGWhiteBG = '\033[1;33;40m'    # 黄色
    HigBlueFGWhiteBG = '\033[1;34;40m'      # 蓝色
    HigFuchsiaFGWhiteBG = '\033[1;35;40m'   # 紫红色
    HigCyanFGWhiteBG = '\033[1;36;40m'      # 青蓝色
    HigWhiteFGWhiteBG = '\033[1;37;40m'     # 白色

    #: no color
    RESET = '\033[0m'      # 终端默认颜色

    def color_str(self, color, s):
        return '{}{}{}'.format(
            getattr(self, color),
            s,
            self.RESET
        )
    #=======================================
    def black(self, s):
        return self.color_str('BLACK', s)
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

     #=======================================
    def higblack(self, s):
        return self.color_str('HigBLACK', s)
    def higred(self, s):
        return self.color_str('HigRED', s)

    def higgreen(self, s):
        return self.color_str('HigGREEN', s)

    def higyellow(self, s):
        return self.color_str('HigYELLOW', s)

    def higblue(self, s):
        return self.color_str('HigBLUE', s)

    def higfuchsia(self, s):
        return self.color_str('HigFUCHSIA', s)

    def higcyan(self, s):
        return self.color_str('HigCYAN', s)

    def higwhite(self, s):
        return self.color_str('HigWHITE', s)

     #=======================================
    def higblackfg_whitebg(self, s):
        return self.color_str('HigBlackFG_WhiteBG', s)
   
    def higredfg_whitebg(self, s):
        return self.color_str('HigRedFG_WhiteBG', s)

    def higgreenfg_whitebg(self, s):
        return self.color_str('HigGreenFGWhiteBG', s)

    def higyellowfg_whitebg(self, s):
        return self.color_str('HigYellowFGWhiteBG', s)

    def higbluefg_whitebg(self, s):
        return self.color_str('HigBlueFGWhiteBG', s)

    def higfuchsiafg_whitebg(self, s):
        return self.color_str('HigFuchsiaFGWhiteBG', s)

    def higcyanfg_whitebg(self, s):
        return self.color_str('HigCyanFGWhiteBG', s)

    def higwhitefg_whitebg(self, s):
        return self.color_str('HigWhiteFGWhiteBG', s)

# # ----------使用示例如下:-------------
# color = ColoPrint()

# print('1==='*20)
# print (color.black(f"黑色   {12}  {12.22}"))
# print (color.red(f"红色    {12}  {12.22} {'jack'}"  ))
# print (color.green(f"绿色   {12}  {12.22} {'jack'}  "))
# print (color.yellow(f"黄色  {12}  {12.22} {'jack'}"))
# print (color.blue(f"蓝色   {12}  {12.22} {'jack'}"))
# print (color.fuchsia( f"紫色  {12}  {12.22} {'jack'}"))
# print (color.cyan(f"青色   {12}  {12.22} {'jack'}"))
# print (color.white(f"白色  {12}  {12.22} {'jack'}"))


# print('2==='*20)
# print (color.higblack(f"高亮黑色   {12}  {12.22}"  ))
# print (color.higred( f"高亮红色   {12}  {12.22}"  ))
# print (color.higgreen( f"高亮绿色   {12}  {12.22}"  ))
# print (color.higyellow( f"高亮黄色   {12}  {12.22}" ))
# print (color.higblue(f"高亮蓝色   {12}  {12.22}"  ))
# print (color.higfuchsia(f"高亮紫色   {12}  {12.22}"  ))
# print (color.higcyan( f"高亮青色   {12}  {12.22}"  ))
# print (color.higwhite( f"高亮白色   {12}  {12.22}"   ))



# print('3==='*20)
# print (color.higblackfg_whitebg( f"高亮黑色  黑底   {12}  {12.22}"  ))
# print (color.higredfg_whitebg( f"高亮红色  黑底   {12}  {12.22}"    ))
# print (color.higgreenfg_whitebg( f"高亮绿色  黑底   {12}  {12.22}"  ))
# print (color.higyellowfg_whitebg( f"高亮黄色  黑底   {12}  {12.22}"  ))
# print (color.higbluefg_whitebg( f"高亮蓝色  黑底   {12}  {12.22}"  ))
# print (color.higfuchsiafg_whitebg( f"高亮紫色  黑底   {12}  {12.22}" ))
# print (color.higcyanfg_whitebg( f"高亮青色  黑底   {12}  {12.22}"  ))
# print (color.higwhitefg_whitebg( f"高亮白色  黑底   {12}  {12.22}"  ))
























