#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:16:14 2023

@author: jack
"""

import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
fonte1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=24)

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Light Nerd Font Complete Mono.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove SemiLight Nerd Font Complete.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Regular Nerd Font Complete Mono.otf", size=20)





import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文


# 可以显示的形状    marker名称
# ϖ	\varpi
# ϱ	\varrho
# ς	\varsigma
# ϑ	\vartheta
# ξ	\xi
# ζ	\zeta
# Δ	\Delta
# Γ	\Gamma
# Λ	\Lambda
# Ω	\Omega
# Φ	\Phi
# Π	\Pi
# Ψ	\Psi
# Σ	\Sigma
# Θ	\Theta
# Υ	\Upsilon
# Ξ	\Xi
# ℧	\mho
# ∇	\nabla
# ℵ	\aleph
# ℶ	\beth
# ℸ	\daleth
# ℷ	\gimel
# /	/
# [	[
# ⇓	\Downarrow
# ⇑	\Uparrow
# ‖	\Vert
# ↓	\downarrow
# ⟨	\langle
# ⌈	\lceil
# ⌊	\lfloor
# ⌞	\llcorner
# ⌟	\lrcorner
# ⟩	\rangle
# ⌉	\rceil
# ⌋	\rfloor
# ⌜	\ulcorner
# ↑	\uparrow
# ⌝	\urcorner
# \vert
# {	\{
# \|
# }	\}
# ]	]
# |
# ⋂	\bigcap
# ⋃	\bigcup
# ⨀	\bigodot
# ⨁	\bigoplus
# ⨂	\bigotimes
# ⨄	\biguplus
# ⋁	\bigvee
# ⋀	\bigwedge
# ∐	\coprod
# ∫	\int
# ∮	\oint
# ∏	\prod
# ∑	\sum

fig, axs = plt.subplots(2, 1, figsize=(8, 13))
#=============================== 0 ===========================================
#常规marker使用
axs[0].plot([1,2,3],[1,2,3],marker=4, markersize=15, color='lightblue',label='常规marker')
axs[0].plot([1.8,2.8,3.8],[1,2,3],marker='2', markersize=15, color='#ec2d7a',label='常规marker')

#非常规marker使用
#注意使用两个$符号包围名称
axs[0].plot([1,2,3],[4,5,6],marker='$\circledR$', markersize=15, color='r', alpha=0.5,label='非常规marker')
axs[0].plot([1.5,2.5,3.5],[1.25,2.1,6.5],marker='$\heartsuit$', markersize=15, color='#f19790', alpha=0.5,label='非常规marker')
axs[0].plot([1,2,3],[2.5,6.2,8],marker='$\clubsuit$', markersize=15, color='g', alpha=0.5,label='非常规marker')

#自定义marker
axs[0].plot([1.2,2.2,3.2],[1,2,3],marker='$666$', markersize=15, color='#2d0c13',label='自定义marker')
axs[0].legend(loc='upper left')


#=============================== 1 ===========================================
#字符型linestyle使用方法
axs[1].plot([1,2,3],[1,2,13],linestyle='dotted', color='#1661ab', linewidth=5, label='字符型线性：dotted')

#元组型lintstyle使用方法
axs[1].plot([0.8,0.9,1.5],[0.8,0.9,21.5],linestyle=(0,(3, 1, 1, 1, 1, 1)), color='#ec2d7a', linewidth=5, label='元组型线性：(0,(3, 1, 1, 1, 1, 1)')



#自定义inestyle
axs[1].plot([1.5,2.5,3.5],[1,2,13],linestyle=(0,(1,2,3,4,2,2)), color='black', linewidth=5, label='自定义线性：(0,(1,2,3,4,2,2)))')
axs[1].plot([2.5,3.5,4.5],[1,2,13],linestyle=(2,(1,2,3,4,2,2)), color='g', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
axs[1].legend()

#=====================================================================================================
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.7)

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5,y=0.96,)

out_fig = plt.gcf()
out_fig.savefig(filepath2+'hhsss.eps', format='eps', bbox_inches='tight')
#out_fig.savefig(filepath2+'hh.svg', format='svg', dpi=1000, bbox_inches='tight')
out_fig.savefig(filepath2+'hhsss.pdf', format='pdf',  bbox_inches='tight')

plt.show()













