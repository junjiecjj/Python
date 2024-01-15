#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1、标记（marker）
# matplotlib入门级marker
# matplotlib一般marker位于matplotlib.lines import Line2D中，共计37种，可以输出来康康有哪些：

# from matplotlib.lines import Line2D
# print([m for m, func in Line2D.markers.items()
# if func != 'nothing' and m not in Line2D.filled_markers] + list(Line2D.filled_markers))
# ['.', ',', '1', '2', '3', '4', '+', 'x', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

# 可以显示的形状    marker名称
# ϖ	\varpi    ★ \bigstar
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
# Ⓢ \circledS

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']

#  ♣ \clubsuit   ⋈ \bowtie  ⊡ \boxdot  ⊠ \boxtimes   ⊟ \boxminus  ⊞ \boxplus  ⨀ \bigodot  ⨁ \bigoplus  ⨂ \bigotimes  ★ \bigstar  ■ \blacksquare  ® \circledR   Ⓢ \circledS  ▽ \bigtriangledown   △ \bigtriangleup  ◀ \blacktriangleleft  ▶ \blacktriangleright   ♢ \diamondsuit  ♡ \heartsuit  ϱ \varrho  ∇ \nabla  Υ \Upsilon  ℧ \mho  Φ \Phi  ⋇ \divideontimes  ⊗ \otimes  ⊙ \odot   ⊖ \ominus  ⊕ \oplus  ⊘ \oslash

# 2 、线型（linestyle）
# 线性 （linestyle）可分为字符串型的元组型的：

# 字符型linestyle
# 有四种，如下：

# linestyle_str = [
# ('solid', 'solid'), # Same as (0, ()) or '-'；solid’， (0, ()) ， '-'三种都代表实线。
# ('dotted', 'dotted'), # Same as (0, (1, 1)) or '.'
# ('dashed', 'dashed'), # Same as '--'
# ('dashdot', 'dashdot')] # Same as '-.'
# 元组型linestyle：
# 直接修改元组中的数字可以呈现不同的线型，所以有无数种该线型。

# linestyle_tuple = [
# ('loosely dotted', (0, (1, 10))),
# ('dotted', (0, (1, 1))),
# ('densely dotted', (0, (1, 2))),
# ('loosely dashed', (0, (5, 10))),
# ('dashed', (0, (5, 5))),
# ('densely dashed', (0, (5, 1))),
# ('loosely dashdotted', (0, (3, 10, 1, 10))),
# ('dashdotted', (0, (3, 5, 1, 5))),
# ('densely dashdotted', (0, (3, 1, 1, 1))),
# ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
# ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
# ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
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


mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_', '$\\clubsuit$', '$\\bowtie$', '$\\boxdot$', '$\\boxtimes$', '$\\boxminus$',  '$\\boxplus$', '$\\bigodot$','$\\bigoplus$', '$\\bigotimes$', '$\\bigstar$', '$\\blacksquare$', '$\circledR$', '$\circledS$',  '$\\bigtriangledown$','$\\bigtriangleup$','$\\blacktriangleleft$', '$\\blacktriangleright$', '$\\diamondsuit',   '$\heartsuit$', '$666$', '$\\varrho$', '$\\Omega$', '$\\nabla$', '$\\Upsilon$', '$\\mho$', '$\\Phi$',  '$\\divideontimes$', '$\\otimes$', '$\\odot$', '$\\ominus$', '$\\oplus$', '$\\oslash$', '$\\otimes$']
color = ['#1E90FF', '#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE', '#00CED1', '#CD5C5C', '#7B68EE', '#0000FF', '#FF0000', '#808000' ]
lsty = ['-', ':', '--', '-.', (0, (1, 1)), (0, (1, 2)), (0, (5, 1)), (0, (1, 10)), (0, (1, 2)),  (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]
alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']



import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文



fig, axs = plt.subplots(2, 1, figsize=(8, 10), constrained_layout = True)
#=============================== 0 ===========================================
#常规marker使用
axs[0].plot([1,2,3],[1,2,3], marker = '4', markersize=15, color='lightblue',label='常规marker')
axs[0].plot([1.8,2.8,3.8],[1,2,3], marker='2', markersize=15, color='#ec2d7a',label='常规marker')

#非常规marker使用
#注意使用两个$符号包围名称
axs[0].plot([1,2,3],[4,5,6], marker='$\circledR$', markersize=15, color='r', alpha=0.5,label='非常规marker')
axs[0].plot([1.5,2.5,3.5],[1.25,2.1,6.5], marker='$\heartsuit$', markersize=15, color='#f19790', alpha=0.5,label='非常规marker')
axs[0].plot([1,2,3],[2.5,6.2,8], marker = '$\\odot$', markersize=15, color='g', alpha=0.5,label='非常规marker')


#自定义marker
axs[0].plot([1.2,2.2,3.2],[1,2,3],marker='$666$', markersize=15, color='#2d0c13',label='自定义marker')
axs[0].legend(loc='upper left')


#=============================== 1 ===========================================
#字符型linestyle使用方法
axs[1].plot([1,2,3],[1,2,13],linestyle='-', color='#1661ab', linewidth=5, label='字符型线性：-')
axs[1].plot([1,3,4],[1,2,13],linestyle='-.', color='#1231ab', linewidth=5, label='字符型线性：-.')
axs[1].plot([2,4,5],[1,2,13],linestyle=':', color='#16abab', linewidth=5, label='字符型线性：:')
axs[1].plot([2,5,6],[1,2,13],linestyle='--', color='#166134', linewidth=5, label='字符型线性：--')


# linestyle_tuple = [
# ('loosely dotted', (0, (1, 10)),
# ('dotted', (0, (1, 1))),
# ('densely dotted', (0, (1, 2)),
# ('loosely dashed', (0, (5, 10)),
# ('dashed', (0, (5, 5)),
# ('densely dashed', (0, (5, 1)),
# ('loosely dashdotted', (0, (3, 10, 1, 10)),
# ('dashdotted', (0, (3, 5, 1, 5)),
# ('densely dashdotted', (0, (3, 1, 1, 1)),
# ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5)),
# ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)),
# ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

#元组型lintstyle使用方法
# axs[1].plot([0.8,0.9,1.5],[0.8,0.9,21.5],linestyle=(0,(3, 1, 1, 1, 1, 1)), color='#ec2d7a', linewidth=5, label='元组型线性：(0,(3, 1, 1, 1, 1, 1)')

# axs[1].plot([1.5,2.5,3.5],[1,2,13],linestyle=(0,(1,2,3,4,2,2)), color='b', linewidth=5, label='自定义线性：(0,(1,2,3,4,2,2)))')

# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (1, 10)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (1, 1)), color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (1, 2)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (5, 10)),  color='b', linewidth=5, label='自定义线性：(0, (5, 10))')
axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (5, 5)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (5, 1)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (3, 10, 1, 10)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (3, 5, 1, 5)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (3, 1, 1, 1)), color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (3, 5, 1, 5, 1, 5)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (3, 10, 1, 10, 1, 10)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (3, 1, 1, 1, 1, 1)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')
# axs[1].plot([1,2,3,4, 5, 6],[1,2,3,6,9,12],linestyle=(0, (3, 5, 1, 5, 1, 5)),  color='b', linewidth=5, label='自定义线性：(1,(1,2,3,4,2,2)))')



axs[1].legend()

#=====================================================================================================
# plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3, hspace=0.7)

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5,y=0.96,)

out_fig = plt.gcf()
out_fig.savefig(filepath2+'hhsss.eps', format='eps', bbox_inches='tight')
#out_fig.savefig(filepath2+'hh.svg', format='svg', dpi=1000, bbox_inches='tight')
# out_fig.savefig(filepath2+'hhsss.pdf', format='pdf',  bbox_inches='tight')

plt.show()













