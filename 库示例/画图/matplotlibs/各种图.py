#!/usr/bin/env python3.6
# -*-coding=utf-8-*-

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

fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
fonttX = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
fonttY = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
fonttitle = {'style': 'normal', 'size': 17}
fontt2 = {'style': 'normal', 'size': 19, 'weight': 'bold'}
fontt3 = {'style': 'normal', 'size': 16, }


'''
linestyle 是设置线头风格，-为实线，--为虚线，:为虚线，-.为点划线, . is dotted, 'o' is dotted
linewidth 设置线条宽度
label=r'$sin(x)$是给曲线打上标签，但是只当一副子图中画出多幅图时有用
marker='*'是设置标志字符, * . o v ^ < > 1 2 3 4 8 s p P h H + x X D d | _
markerfacecolor='red'是设置标志颜色
markersize=12是设置标志大小
https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
1、颜色、点标记与线型设置
1）常用的参数名：小括号中都是简写
color ：线条颜色。
linestyle(ls)：线条形状。
linewidth(lw)：线宽。
marker：点标记形状。
markersize(ms)：点标记的大小。
markeredgecolor(mec)：点边缘颜色。
markeredgewidth(mew)：点边缘宽度。
markerfacecolor(mfc)：点的颜色。

ax1.set_title('sin(x)')给子图打上标题
ax2.annotate()是给子图中某处打上箭头并给出描述
plt.suptitle('cos and sin')给整个画布打上大标题
ax.set_xticks([-4, -2, 0, 2])设置刻度值
ax.tick_params(labelcolor='r', labelsize='medium', width=3)设置刻度的大小颜色宽度等，但是不能设置刻度字体


marker 可选参数： https://www.runoob.com/matplotlib/matplotlib-marker.html
'.'       point marker
','       pixel marker
'o'       circle marker
'v'       triangle_down marker
'^'       triangle_up marker
'<'       triangle_left marker
'>'       triangle_right marker
'1'       tri_down marker
'2'       tri_up marker
'3'       tri_left marker
'4'       tri_right marker
'8'       octagon
's'       square marker
'p'       pentagon marker
'*'       star marker
'h'       hexagon1 marker
'H'       hexagon2 marker
'+'       plus marker
'x'       x marker
'D'       diamond marker
'd'       thin_diamond marker
'|'       vline marker
'_'       hline marker

可用颜色：
cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkgreen':            '#4ea142',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#28a428',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#bc6035',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}


Matplotlib savefig()的语法
savefig(fname, dpi=None, facecolor=’w’, edgecolor=’w’, orientation=’portrait’, papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
参数
• fname: （字符串或者仿路径或仿文件）如果格式已经设置，这将决定输出的格式并将文件按fname来保存。如果格式没有设置，在fname有扩展名的情况下推断按此保存，没有扩展名将按照默认格式存储为“png”格式，并将适当的扩展名添加在fname后面。
• dpi: 分辨率，每英寸的点数
• facecolor（颜色或“auto”，默认值是“auto”）：图形表面颜色。如果是“auto”，使用当前图形的表面颜色。
• edgecolor（颜色或“auto”，默认值：“auto”）：图形边缘颜色。如果是“auto”，使用当前图形的边缘颜色。
• orientation – {‘landscape,’ ‘portrait’}: 目前只有后端支持。.
• format（字符串）：文件格式，比如“png”，“pdf”，“svg”等，未设置的行为将被记录在fname中。
• papertype: papertypes可以设置为“a0到a10”， “executive,” “b0 to b10”, “letter,” “legal,” “ledger.”
• bbox_inches: 只有图形给定部分会被保存。设置为“tight”用以恰当的匹配所保存的图形。
• pad_inches: (默认: 0.1)所保存图形周围的填充量。
• transparent: 用于将图片背景设置为透明。图形也会是透明，除非通过关键字参数指定了表面颜色和/或边缘颜色。
返回类型
matplotlib savefig()函数将所绘制的图形保存在本地计算机.




matplotlib.pyplot.tick_params参数
axis——轴：{ ’ x ’ ，’ y ’ ，’ both ’ }
参数axis的值分别代表设置X轴、Y轴以及同时设置。默认值为 ’ both ’ 。
reset——重置：布尔
如果为True，则在处理其他关键字参数之前将所有参数设置为默认值。默认值为False。
which——其中：{ ‘ major ’ ，‘ minor ’ ，‘ both ’ }
参数which的值分别代表为“主要”，“次要”，“两者”。默认值为’ major '。
direction / tickdir——方向： {‘in’，‘out’，‘inout’}
将刻度线置于轴内，轴外或两者皆有。
size / length——长度：浮点
刻度线长度（以磅为单位）。
width——宽度：浮动
刻度线宽度（以磅为单位）。
color——颜色：颜色
刻度颜色；接受任何mpl颜色规格。
pad ——垫：浮球
刻度和标签之间的距离（以磅为单位）。
labelsize——标签大小：float 或 str
刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
labelcolor——标签颜色：颜色
刻度标签颜色；mpl颜色规格
colors——颜色：颜色
将刻度颜色和标签颜色更改为相同的值：mpl color spec。
zorder——zorder：浮动
勾选并标记zorder。
bottom，top，left，right——底部，顶部，左侧，右侧：布尔
是否绘制各个刻度。
labelbottom，labeltop，labelleft，labelright——标签底部，标签顶部，标签左侧，标签右侧：布尔
是否绘制各个刻度标签。
labelrotation：浮动
刻度线标签逆时针旋转给定的度数
gridOn——网格线：布尔
是否添加网格线
grid_color——网格线颜色：颜色
将网格线颜色更改为给定的mpl颜色规格。
grid_alpha——网格线透明度：浮点数
网格线的透明度：0（透明）至1（不透明）。
grid_linewidth——网格线宽度：浮点数
网格线的宽度（以磅为单位）。
grid_linestyle——网格线型：字符串
任何有效的Line2D线型规范。


matplotlib.pyplot.tick_params()用于更改刻度、刻度标签和网格线的外观。

语法:

matplotlib.pyplot.tick_params(axis='both', **kwargs)
PythonCopy
参数:

Parameter	Value	Use
axis	{‘ x ‘， ‘ y ‘， ‘ both ‘}，可选	将参数应用于哪个轴。默认设置是“两个”。
reset	bool，默认:False	如果为True，在处理其他关键字参数之前，将所有参数设置为默认值。
which	{‘major’, ‘minor’, ‘both’}	默认是“主要”;应用勾号的参数。
direction	{‘in’, ‘out’, ‘inout’}	将刻度置于坐标轴内、轴外，或同时置于两者。
length	float	以点为单位的滴答长度。
width	float	默认是“主要”;应用勾号的参数。
color	颜色	蜱虫的颜色。
pad	float	标记和标签之间的点距离。
labelsize	float或str	标记的字体大小以点表示或以字符串形式表示(e.g;,“大”)。



matplotlib 绘制曲线时如果数据点较多， 添加 marker 后会出现 marker 重叠或太密集的现象， 可以用 markevery 来控制 marker 的间距。

plt.plot(x, y, marker='o', markevery=10)
markevery 可以设置整数，也可以设为数组格式。

'''


############################################################################################
t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1.1, 0.1)
s4 = np.arcsin(t1)



fig, axs = plt.subplots(4, 1, figsize=(10, 16))
############################################## 0 #############################################

axs[0].plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',  marker='o', markevery=10)
axs[0].plot(t, s3, color='r', linestyle='-', label='tan(x)',)
axs[0].axvline(x=1, ymin=0.4, ymax=0.6, ls='-', linewidth=4, color='b', label='tan(x)',)


font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
font2  = {'family':'Times New Roman','style':'normal','size':17, 'color':'#00FF00'}
axs[0].set_xlabel(r'time (s)时间', fontproperties=font1, fontdict = font2, labelpad=12.5) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。)
axs[0].set_ylabel(r'值(0-1)', fontproperties=font1, fontdict = font2)
axs[0].set_title('sin and tan 函数', fontproperties=font1)

font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
# font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[0].set_title('杰克', loc='left', color='#0000FF', fontproperties=font1)
axs[0].set_title('rose', loc='right', color='#9400D3', fontproperties=font1)
# # 设置 y 就在轴方向显示网格线
axs[0].grid(axis='x', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
axs[0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, labelcolor = "red", colors='blue', rotation=25,)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
axs[0].spines['left'].set_color('m') ## 设置边框线颜色
axs[0].spines['bottom'].set_color('red') ## 设置边框线颜色
axs[0].spines['bottom'].set_linestyle("--") ## 设置边框线线型

axs[0].set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
axs[0].set_ylim(-3, 3)  #拉开坐标轴范围显示投影

x_major_locator=MultipleLocator(0.2)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(1)
#把y轴的刻度间隔设置为10，并存在变量里

axs[0].xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
axs[0].yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数

############################################# 1 ###############################################
font  = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

axs[1].plot(t, s2, color='r', linestyle='--', label=r'$cos(x)$',)
axs[1].axvline(x=1, ls=':', color='b', label='disruption')
axs[1].axhline(y=0.5, ls=':', color='r', label='阈值')
axs[1].annotate(r'$t_{real\_disr}$', xy=(1, -1), xytext=(1.1, -0.7), arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), fontproperties=font)


#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

axs[1].set_xlabel(r'time (s)', fontproperties=font2, fontdict = font1, labelpad=12.5) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs[1].set_ylabel(r'值(0-1)', fontproperties=font2, fontdict = font1, labelpad=12.5)
axs[1].set_title('cos(x)', fontproperties=font2)
axs[1].set_title(r'$\mathrm{t}_\mathrm{disr}$=%d' % 1, loc='left', fontproperties=font2,  )
axs[1].grid()


font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

x_major_locator=MultipleLocator(1)               #把x轴的刻度间隔设置为1，并存在变量里
axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[1].tick_params(direction='in', axis='both', top=True, right=True,labelsize=16, width=3,)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


#################################### 2 #############################################
font  = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

axs[2].plot(t1, s4, color='blue', linestyle='-', linewidth=1, label='arcsin()')
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[2].set_xlabel(r'time (s)', fontproperties=font3)
axs[2].set_ylabel(r'值(0-1)', fontproperties=font3)
axs[2].set_title('arcsin(x)', fontproperties=font3)
axs[2].axvline(x=0, ymin=0.4, ymax=0.6, ls='-', linewidth=4, color='b',)
axs[2].annotate(r'$t_{disr}$',
                xy=(0, 0), xycoords='data',
                xytext=(0.4, 0.3), textcoords='figure fraction',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), fontproperties=font)

axs[2].grid()

font3 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font3,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[2].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3,)
labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[2].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[2].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[2].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[2].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
# ##############################################################################################

#################################### 3 #############################################
font  = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

axs[3].plot(t1, s4, color='blue', linestyle='-', linewidth=1, label='arcsin()')
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[3].set_xlabel(r'time (s)', fontproperties=font3)
axs[3].set_ylabel(r'值(0-1)', fontproperties=font3)
axs[3].set_title('arcsin(x)', fontproperties=font3)
axs[3].axvline(x=0, ymin=0.4, ymax=0.6, ls='-', linewidth=4, color='b',)
axs[3].annotate(r'$t_{disr}$',
                xy=(0, 0), xycoords='data',
                xytext=(0.4, 0.1), textcoords='figure fraction',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), fontproperties=font)

axs[3].grid()

font3 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font3 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
legend1 = axs[3].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font3,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
axs[3].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3,)
labels = axs[3].get_xticklabels() + axs[3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[3].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[3].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[3].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[3].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

# ##############################################################################################


#fig.subplots_adjust(hspace=0.6)  # 调节两个子图间的距离
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.7)

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5,y=0.96,)

out_fig = plt.gcf()
out_fig.savefig(filepath2+'hh11.eps', format='eps',dpi=1000, bbox_inches='tight')
#out_fig.savefig(filepath2+'hh.svg', format='svg', dpi=1000, bbox_inches='tight')
out_fig.savefig(filepath2+'hh.pdf', format='pdf', dpi=1000, bbox_inches='tight')
#out_fig.savefig(filepath2+'hh.emf',format='emf',dpi=1000, bbox_inches = 'tight')
#out_fig.savefig(filepath2+'hh.jpg',format='jpg',dpi=1000, bbox_inches = 'tight')
#out_fig.savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



#========================================================================================================================
#===========================================  调整图与图之间的间距 ===============================================
#========================================================================================================================
t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1, 0.1)
s4 = np.arcsin(t1)

fig, axs = plt.subplots(2, 2, figsize=(16, 16) ) # ,constrained_layout=True
# plt.subplots(constrained_layout=True)的作用是:自适应子图在整个figure上的位置以及调整相邻子图间的间距，使其无重叠。
#=============================================== 0 ======================================================
axs[0,0].plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font3  = {'family':'Times New Roman','style':'normal','size':22}
axs[0,0].set_xlabel(r'time (s)', fontproperties=font3)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
axs[0,0].set_ylabel(r'值(sin(x))', fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':22}
axs[0,0].set_title('sin(x)', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0,0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
axs[0,0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, labelcolor = "red", colors='blue', rotation=25,)
# axs[0,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=26, width=3,)
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[0,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[0,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[0,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[0,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
axs[0,0].spines['bottom'].set_color('red') ## 设置边框线颜色
axs[0,0].spines['bottom'].set_linestyle("--") ## 设置边框线线型

xlabels = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
axs[0,0].set_xticklabels(xlabels)
#=============================================== 1 ======================================================
axs[0,1].plot(t, s2, color='r', linestyle='-', label='cos(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=32)
font   = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#00FF00'}
axs[0,1].set_xlabel(r'time (s)',  fontdict = font2)
axs[0,1].set_ylabel(r'值(cos(x))', fontproperties=font3, fontdict = font2)
axs[0,1].set_title('cos(x)', fontproperties=font3)

axs[0,1].grid(axis='x', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[0,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[0,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[0,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[0,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

axs[0,1].set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
axs[0,1].set_ylim(-1.1, 1.2)  #拉开坐标轴范围显示投影

x_major_locator=MultipleLocator(0.2)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.25)
#把y轴的刻度间隔设置为10，并存在变量里

axs[0,1].xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
axs[0,1].yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数

#=============================================== 2 ======================================================
axs[1,0].plot(t, s3, color='g', linestyle='-', label='tan(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,0].set_xlabel(r'time (s)', fontproperties=font3)
axs[1,0].set_ylabel(r'值(tan(x))', fontproperties=font3)
axs[1,0].set_title('tan(x)', fontproperties=font3)

axs[1,0].grid(  color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[1,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

#=============================================== 3 ======================================================
axs[1,1].plot(t, s4, color='b', linestyle='-', label='arcsin(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#9400D3'}
axs[1,1].set_xlabel(r'time (s)', fontproperties=font3)
axs[1,1].set_ylabel(r'值(arcsin(x))', fontproperties=font3, fontdict = font2)
axs[1,1].set_title('arcsin(x)', fontproperties=font3)

axs[1,1].grid(axis='y', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[1,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


#=====================================================================================
# add_axes新增子区域
# 定义数据
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]
#新增区域ax2,嵌套在ax1内
left, bottom, width, height = 0.8, 0.15, 0.15, 0.15
# 获得绘制的句柄
axss =  fig.add_axes([left, bottom, width, height])
axss.plot(x, y, 'b')
#=====================================================================================


#=====================================================================================
# plt.cla() # plt.cla()清除轴 ，即当前图中的当前活动轴。 它使其他轴保持不变。

# plt.clf() # plt.clf()使用其所有轴清除整个当前图形 ，但使窗口保持打开状态，以便可以将其重新用于其他绘图。

# fig.clf() # 清除整个图
#=====================================================================================
# plt.cla() # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
# plt.clf() # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。
# plt.close() # 关闭 window，如果没有指定，则指当前 window。

#=====================================================================================
#fig.tight_layout(pad=6, h_pad=4, w_pad=4)
# matplotlib.pyplot.tight_layout(*, pad=1.08, h_pad=None, w_pad=None, rect=None)
# pad,h_pad,w_pad分别调整子图和figure边缘，以及子图间的相距高度、宽度。

# 调节两个子图间的距离
# plt.subplots_adjust(left=None,bottom=None,right=None,top=0.85,wspace=0.1,hspace=0.1)
# 有六个可选参数来控制子图布局。值均为0~1之间。其中left、bottom、right、top围成的区域就是子图的区域。 left, right, bottom, top：子图所在区域的边界。 当值大于1.0的时候子图会超出figure的边界从而显示不全；值不大于1.0的时候，子图会自动分布在一个矩形区域（下图灰色部分）。要保证left < right, bottom < top，否则会报错。
# wspace、hspace分别表示子图之间左右、上下的间距。
# wspace, hspace：子图之间的横向间距、纵向间距分别与子图平均宽度、平均高度的比值。实际的默认值由matplotlibrc文件控制的。
plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.2, hspace=0.3)

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5, y=0.99,)


out_fig = plt.gcf()

filepath2 = '/home/jack/snap/'
# out_fig.savefig(filepath2+'hh.pdf', format='pdf', dpi=1000, bbox_inches='tight')
#out_fig .savefig(filepath2+'hh.emf',format='emf',dpi=1000, bbox_inches = 'tight')
out_fig .savefig(filepath2+'hh22.eps',format='eps',dpi=1000, bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()




#=====================================================================================================

"""
plt绘图之自适应子图间距的三种方式

(一)
plt.subplots(constrained_layout=True)的作用是:
    constrained_layout 会自动调整子图和装饰，使其尽可能地适合图中。
    自适应子图在整个figure上的位置以及调整相邻子图间的间距，使其无重叠。
    constrained_layout 自动调整子批次和装饰，如图例和颜色栏，以便它们适合图形窗口，同时尽可能保留用户请求的逻辑布局。
    constrained_layout 类似于 tight_layout ，但使用约束解算器来确定允许它们匹配的轴的大小。

(二)：
fig.tight_layout(pad=5, h_pad=1,w_pad=1)
matplotlib.pyplot.tight_layout(*, pad=1.08, h_pad=None, w_pad=None, rect=None)
pad, h_pad, w_pad分别调整子图和figure边缘，以及子图间的相距高度、宽度。

tight_layout() 方法会自动保持子图之间的正确间距。

函数的参数如下：

pad：所有子图整体边缘相对于图像边缘的内边距，距离单位为字体大小的比例（小数，与rcParams["font.size"]相关）。可选参数。浮点数。默认值为1.08。
h_pad, w_pad：子图之间的内边距，距离单位为字体大小的比例（小数）。可选参数。浮点数。默认值为pad。
rect：绘制子图的矩形区域的归一化坐标。可选参数。4元组(left, bottom, right, top)。默认值为(0, 0, 1, 1)。


(三)： 我们可以使用 plt.subplots_adjust() 方法来更改子图之间的间距。 https://blog.csdn.net/ggt55ng6/article/details/88879689
plt.subplots_adjust(left=None,bottom=None,right=None,top=0.85,wspace=0.1,hspace=0.1)
    有六个可选参数来控制子图布局。值均为0~1之间。其中left、bottom、right、top围成的区域就是子图的区域。 left, right, bottom, top：子图所在区域的边界。 当值大于1.0的时候子图会超出figure的边界从而显示不全；值不大于1.0的时候，子图会自动分布在一个矩形区域（下图灰色部分）。要保证left < right, bottom < top，否则会报错。
    wspace、hspace分别表示子图之间左右、上下的间距。
    wspace, hspace：子图之间的横向间距、纵向间距分别与子图平均宽度、平均高度的比值。实际的默认值由matplotlibrc文件控制的。
    plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.5, hspace=0.2)

    wspace 和 hspace 指定子图之间保留的空间。它们分别是轴的宽度和高度的分数。
    left，right，top 和 bottom 参数指定了子图的四个边的位置。它们是图形的宽度和高度的比例。

    函数的参数如下：

    left：所有子图整体相对于图像的左外边距，距离单位为图像宽度的比例（小数）。可选参数。浮点数。默认值为0.125。
    right：所有子图整体相对于图像的右外边距，距离单位为图像宽度的比例（小数）。可选参数。浮点数。默认值为0.0。
    bottom：所有子图整体相对于图像的下外边距，距离单位为图像高度的比例（小数）。可选参数。浮点数。默认值为0.11。
    top：所有子图整体相对于图像的上外边距，距离单位为图像高度的比例（小数）。可选参数。浮点数。默认值为0.88。
    wspace：子图间宽度内边距，距离单位为子图平均宽度的比例（小数）。浮点数。默认值为0.2。
    hspace：子图间高度内边距，距离单位为子图平均高度的比例（小数）。可选参数。浮点数。默认值为0.2。


(4) plt.subplot_tool() 方法更改 Matplotlib 子图大小和间距
上述三者用一个调节即可




Matplotlib savefig()的语法
savefig(fname, dpi=None, facecolor=’w’, edgecolor=’w’, orientation=’portrait’, papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
参数
• fname: （字符串或者仿路径或仿文件）如果格式已经设置，这将决定输出的格式并将文件按fname来保存。如果格式没有设置，在fname有扩展名的情况下推断按此保存，没有扩展名将按照默认格式存储为“png”格式，并将适当的扩展名添加在fname后面。
• dpi: 分辨率，每英寸的点数
• facecolor（颜色或“auto”，默认值是“auto”）：图形表面颜色。如果是“auto”，使用当前图形的表面颜色。
• edgecolor（颜色或“auto”，默认值：“auto”）：图形边缘颜色。如果是“auto”，使用当前图形的边缘颜色。
• orientation – {‘landscape,’ ‘portrait’}: 目前只有后端支持。.
• format（字符串）：文件格式，比如“png”，“pdf”，“svg”等，未设置的行为将被记录在fname中。
• papertype: papertypes可以设置为“a0到a10”， “executive,” “b0 to b10”, “letter,” “legal,” “ledger.”
• bbox_inches: 只有图形给定部分会被保存。设置为“tight”用以恰当的匹配所保存的图形。
• pad_inches: (默认: 0.1)所保存图形周围的填充量。
• transparent: 用于将图片背景设置为透明。图形也会是透明，除非通过关键字参数指定了表面颜色和/或边缘颜色。
返回类型
matplotlib savefig()函数将所绘制的图形保存在本地计算机.

"""
#=====================================================================================================

"""
subplot和subplots区别，用法:
 plt.figure的作用是定义一个大的图纸，可以设置图纸的大小、分辨率等，例如

fig = plt.figure(figsize=(16,16),dpi=300)  # 初始化一张画布
plt.plot() 是直接在当前活跃的的axes上面作图，注意是当前活跃的

知道这两点基础知识后，再来看subplot和subplots

(一) plt.subplot:
fig = plt.figure(figsize=(12, 4), dpi=200)
for i in range(len(img)):
    plt.subplot(1, len(img),i+1)
    plt.imshow(img[i])
plt.show()

plt.subplot作用是指定子图的位置，比如说现在总共有1行10列，当前子图位于哪里；
使用这个函数时需要先定义一个大的图纸，因为subplot函数无法更改图纸的大小和分辨率等信息；所以必须通过fig = plt.figure(figsize=(12, 4), dpi=200)来定义图纸相关设置；
同时，后续对于这个函数便捷的操作就是直接用plt，获取当前活跃的图层

(二)  plt.subplots
fig, ax = plt.subplots(1, len(img), figsize=(15, 10))
for i in range(len(img)):
    ax[i].imshow(img[i])
plt.show()

使用plt.subplots函数时，可以直接在该函数内部设置子图纸信息
该函数返回两个变量，一个是Figure实例fig，另一个 AxesSubplot实例ax 。fig代表整个图像，ax代表坐标轴和画的子图，通过下标获取需要的子区域。
后续我们需要对子图操作时，直接ax[i].imshow(img[i])就行

与subplot的区别在于：（1）不需要通过plt来操作图层，每一个图层都有指定的axes；（2）一个写在for循环外面，一个写在里面；归根于原因还是suplots绘制多少图已经指定了，所以ax提前已经准备好了，而subplot函数调用一次就绘制一次，没有指定

subplot和subplots都可以实现画子图功能，只不过subplots帮我们把画板规划好了，返回一个坐标数组对象，而subplot每次只能返回一个坐标对象，subplots还可以直接指定画板的大小。

(三) 坐标轴修改:
我们通常需要修改坐标轴大小、刻度等信息，不论是子图还是一张大图；下面来看一下subplots和subplot在修改坐标方面的差异

plt在修改坐标时直接写plt.xlabel();plt.ylabel();plt.xlim();plt.ylim()等等就行，但是axes和plt不一样，axes需要加上set，例如：axes.set_xlabel();axes.set_xlim() 这一点需要格外注意
对于修改子图的坐标轴信息，很明显是subplots更方便，因为他有独立的axes，更方便让每一个子图的坐标轴不同，例如


"""





"""


##################################################################
#这一段代码和上面一段作用一样
#from __future__ import (absolute_import, division,print_function, unicode_literals)
import matplotlib.pyplot as plt
import numpy as np

t=np.arange(0,2,0.1)
s1=np.sin(2*np.pi*t)
s2=np.cos(2*np.pi*t)
s3=np.tan(2*np.pi*t)

fig=plt.figure()
ax0=fig.add_subplot(211)
ax0.plot(t,s1,color='b',linestyle='-',linewidth=2,marker='*',markerfacecolor='red',markersize=12,label='sin(x)')
ax0.plot(t,s3,color='b',linestyle='-',linewidth=2,marker='o',markerfacecolor='c',markersize=12,label='tan(x)')
ax0.axvline(x=1,ls='--',color='b',label='disr')
ax0.legend(loc='best',shadow=True)
ax0.set_xlabel(r'time (s)')
ax0.set_title('sin and tan')

ax1=fig.add_subplot(212)
ax1.plot(t,s2,color='r',linestyle='--',linewidth=2,markerfacecolor='k',markersize=12,label='cos(x)')
ax1.axvline(x=1,ls=':',color='b',label='disruption')
ax1.annotate(r'$t_{disr}$',xy=(1,-1),xytext=(1.1,-0.7),\
arrowprops=dict(arrowstyle='->',connectionstyle='arc3'))
ax1.legend(loc='best',shadow=True)
ax1.set_xlabel(r'time (s)')
ax1.set_title('cos(x)')

fig.subplots_adjust(hspace=0.6)#调节两个子图间的距离

plt.suptitle('cos and sin')
plt.show()

################################################

"""


#===============================================================================
#   多子图的方式一
#===============================================================================

"""
Matplotlib 绘制多图
我们可以使用 pyplot 中的 subplot() 和 subplots() 方法来绘制多个子图。

subplot() 方法在绘图时需要指定位置，subplots() 方法可以一次生成多个，在调用时只需要调用生成对象的 ax 即可。

subplot
subplot(nrows, ncols, index, **kwargs)
subplot(pos, **kwargs)
subplot(**kwargs)
subplot(ax)
以上函数将整个绘图区域分成 nrows 行和 ncols 列，然后从左到右，从上到下的顺序对每个子区域进行编号 1...N ，左上的子区域的编号为 1、右下的区域编号为 N，编号可以通过参数 index 来设置。

设置 numRows ＝ 1，numCols ＝ 2，就是将图表绘制成 1x2 的图片区域, 对应的坐标为：

(1, 1), (1, 2)
plotNum ＝ 1, 表示的坐标为(1, 1), 即第一行第一列的子图。

plotNum ＝ 2, 表示的坐标为(1, 2), 即第一行第二列的子图。
"""


import matplotlib.pyplot as plt
import numpy as np

#plot 1:
x = np.array([0, 6])
y = np.array([0, 100])

plt.subplot(2, 2, 1)
plt.plot(x,y)
plt.title("plot 1")

#plot 2:
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

plt.subplot(2, 2, 2)
plt.plot(x,y)
plt.title("plot 2")

#plot 3:
x = np.array([1, 2, 3, 4])
y = np.array([3, 5, 7, 9])

plt.subplot(2, 2, 3)
plt.plot(x,y)
plt.title("plot 3")

#plot 4:
x = np.array([1, 2, 3, 4])
y = np.array([4, 5, 6, 7])

plt.subplot(2, 2, 4)
plt.plot(x,y)
plt.title("plot 4")

plt.suptitle("RUNOOB subplot Test")
plt.show()

#===============================================================================
#   多子图的方式2
#===============================================================================
"""
subplots()
subplots() 方法语法格式如下：

matplotlib.pyplot.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)参数说明：
nrows：默认为 1，设置图表的行数。
ncols：默认为 1，设置图表的列数。
sharex、sharey：设置 x、y 轴是否共享属性，默认为 false，可设置为 'none'、'all'、'row' 或 'col'。 False 或 none 每个子图的 x 轴或 y 轴都是独立的，True 或 'all'：所有子图共享 x 轴或 y 轴，'row' 设置每个子图行共享一个 x 轴或 y 轴，'col'：设置每个子图列共享一个 x 轴或 y 轴。
squeeze：布尔值，默认为 True，表示额外的维度从返回的 Axes(轴)对象中挤出，对于 N*1 或 1*N 个子图，返回一个 1 维数组，对于 N*M，N>1 和 M>1 返回一个 2 维数组。如果设置为 False，则不进行挤压操作，返回一个元素为 Axes 实例的2维数组，即使它最终是1x1。
subplot_kw：可选，字典类型。把字典的关键字传递给 add_subplot() 来创建每个子图。
gridspec_kw：可选，字典类型。把字典的关键字传递给 GridSpec 构造函数创建子图放在网格里(grid)。
**fig_kw：把详细的关键字参数传给 figure() 函数。
"""
import matplotlib.pyplot as plt
import numpy as np

# 创建一些测试数据 -- 图1
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)


fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')
plt.show()


# 创建两个子图 -- 图2
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)
plt.show()

# 创建四个子图 -- 图3
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, subplot_kw=dict(projection="polar"))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

## 共享 x 轴
# plt.subplots(2, 2, sharex='col')
## 共享 y 轴
# plt.subplots(2, 2, sharey='row')
## 共享 x 轴和 y 轴
# plt.subplots(2, 2, sharex='all', sharey='all')
## 这个也是共享 x 轴和 y 轴
# plt.subplots(2, 2, sharex=True, sharey=True)
plt.show()


# 创建标识为 10 的图，已经存在的则删除
fig, ax = plt.subplots(num=10, clear=True)

plt.show()


#====================================================================================
#  「方式一：通过plt的subplot」
#====================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

filepath2 = '/home/jack/snap/'

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


fig = plt.figure(figsize=(5, 5), dpi=200)

#===================================================================
# 画第1个图：折线图
#===================================================================
x=np.arange(1,100)
plt.subplot(221)
plt.plot(x,x*x, label='减摇控制前')

# 坐标轴的起始点
plt.xlim(0,110)   # xlim: 设置x、y坐标轴的起始点（从哪到哪）
plt.ylim(0,11000) # ylim： 设置x、y坐标轴的起始点（从哪到哪）

# 设置坐标轴刻度
font3  = {'family':'Times New Roman','style':'normal','size':12}
plt.xticks([0,50,90], fontproperties=font3) #  xticks： 设置坐标轴刻度的字体大小
plt.yticks(  fontproperties=font3)          #  yticks： 设置坐标轴刻度的字体大小



ax=plt.gca()#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1); ###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1);   ###设置上部坐标轴的粗细

# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
# pad  刻度距离坐标轴的距离调整
plt.tick_params(direction='in', axis='both',top=True, right=True, labelcolor='b', color = 'r', labelsize=10, width=1, rotation=25, pad = 1)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(10) for label in labels] #刻度值字号

# xlabels = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
# ax.set_xticklabels(xlabels)


# xlabel, ylabel 标签设置
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
plt.xlabel("x test", fontproperties=font2) #  xlabel: 设置横轴、纵轴标签及大小
plt.ylabel("Ysssssssssssssssss\nssssssssssssssssss", fontproperties=font2, linespacing = 0.8, labelpad = 0.3)      #  ylabel: 设置横轴、纵轴标签及大小

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 6)
legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明



# 标题
fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 14}
plt.title("csdn test", fontproperties=fontt, color='#0000FF',)  #   title：设置图片的标题


#===================================================================
# 画第2个图：散点图
#===================================================================

plt.subplot(222)
plt.scatter(np.arange(0,10), np.random.rand(10), label = "scatter")

# # 坐标轴的起始点
# plt.xlim(0,110)   # xlim: 设置x、y坐标轴的起始点（从哪到哪）
# plt.ylim(0,11000) # ylim： 设置x、y坐标轴的起始点（从哪到哪）

# # 设置坐标轴刻度
# font3  = {'family':'Times New Roman','style':'normal','size':12}
# plt.xticks([0,50,90], fontproperties=font3) #  xticks： 设置坐标轴刻度的字体大小
# plt.yticks(  fontproperties=font3)          #  yticks： 设置坐标轴刻度的字体大小


ax=plt.gca()#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);  ###设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1); ###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1);   ###设置上部坐标轴的粗细

# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
plt.tick_params(direction='in', axis='both',top=True, right=True, labelcolor='r', color = 'g', labelsize=10, width=1, rotation=25, pad = 0.01)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(10) for label in labels] #刻度值字号

# xlabel, ylabel
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
plt.xlabel("X", fontproperties=font2) #  xlabel: 设置横轴、纵轴标签及大小
plt.ylabel("Y", fontproperties=font2)      #  ylabel: 设置横轴、纵轴标签及大小

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 6)
legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


# 标题
fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 14}
plt.title("csdn", fontproperties=fontt, color='#000FFF',)  #   title：设置图片的标题

#===================================================================
# 画第3个图：饼图
#===================================================================
plt.subplot(223)
plt.pie( x=[15,30,45,10], labels=list('ABCD'), autopct='%.0f', explode=[0,0.05,0,0], )


#===================================================================
# 画第4个图：条形图
#===================================================================

plt.subplot(224)
plt.bar([20,10,30,25,15],[25,15,35,30,20],color='b')
font3  = {'family':'Times New Roman','style':'normal','size':12}
plt.xticks( fontproperties=font3)
plt.yticks( fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
plt.xlabel("X", fontproperties=font2)
plt.ylabel("Y", fontproperties=font2)
fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
plt.title("csdn", fontproperties=fontt,)


#====================================================
fontt = FontProperties(fname=fontpath+"simsun.ttf", size=12)
plt.suptitle("设置图片的标题", fontproperties=fontt, x=0.5, y=1,)
plt.subplots_adjust(left=0, bottom=0, right=1, top=0.9 , wspace=0.4, hspace=0.4)
plt.show()



#====================================================================================
#  「方式二：通过figure的add_subplot」
#====================================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

filepath2 = '/home/jack/snap/'

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


fig = plt.figure(figsize=(5, 5), dpi = 200)
#===================================================================
# 画第1个图：折线图
#===================================================================

x=np.arange(1,100)
ax1 = fig.add_subplot(221)
ax1.plot(x, x*x, label = "y=x^2")

# 坐标轴的起始点
ax1.set_xlim(0,110)   # xlim: 设置x、y坐标轴的起始点（从哪到哪）
ax1.set_ylim(0,11000) # ylim： 设置x、y坐标轴的起始点（从哪到哪）

# 设置坐标轴刻度
font3  = {'family':'Times New Roman','style':'normal','size':12}
ax1.set_xticks([0,50,90], fontproperties=font3) #  xticks： 设置坐标轴刻度的字体大小
# ax1.set_yticks(  fontproperties=font3)          #  yticks： 设置坐标轴刻度的字体大小


ax1.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax1.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax1.spines['right'].set_linewidth(1); ###设置右边坐标轴的粗细
ax1.spines['top'].set_linewidth(1);   ###设置上部坐标轴的粗细


# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
# pad  刻度距离坐标轴的距离调整
ax1.tick_params(direction='in', axis='both',top=True, right=True, labelcolor='b', color = 'r', labelsize=10, width=1, rotation=25, pad = 1)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(10) for label in labels] #刻度值字号

# xlabel, ylabel 标签设置
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
ax1.set_xlabel("x test", fontproperties=font2) #  xlabel: 设置横轴、纵轴标签及大小
ax1.set_ylabel("Ysssssssssssssssss\nssssssssssssssssss", fontproperties=font2, linespacing = 0.8, labelpad = 0.3)      #  ylabel: 设置横轴、纵轴标签及大小

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 12)
legend1 = ax1.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# 标题
fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 14}
ax1.set_title("csdn test", fontproperties=fontt, color='#0000FF',)  #   title：设置图片的标题


#===================================================================
# 画第2个图：散点图
#===================================================================
ax2=fig.add_subplot(222)
ax2.scatter(np.arange(0,10), np.random.rand(10), label='scatter')


# # 坐标轴的起始点
# plt.xlim(0,110)   # xlim: 设置x、y坐标轴的起始点（从哪到哪）
# plt.ylim(0,11000) # ylim： 设置x、y坐标轴的起始点（从哪到哪）

# # 设置坐标轴刻度
# font3  = {'family':'Times New Roman','style':'normal','size':12}
# plt.xticks([0,50,90], fontproperties=font3) #  xticks： 设置坐标轴刻度的字体大小
# plt.yticks(  fontproperties=font3)          #  yticks： 设置坐标轴刻度的字体大小



ax2.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax2.spines['left'].set_linewidth(1);  ###设置左边坐标轴的粗细
ax2.spines['right'].set_linewidth(1); ###设置右边坐标轴的粗细
ax2.spines['top'].set_linewidth(1);   ###设置上部坐标轴的粗细

# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
ax2.tick_params(direction='in', axis='both',top=True, right=True, labelcolor='r', color = 'g', labelsize=10, width=1, rotation=25, pad = 0.01)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(10) for label in labels] #刻度值字号

# xlabel, ylabel
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
ax2.set_xlabel("X", fontproperties=font2) #  xlabel: 设置横轴、纵轴标签及大小
ax2.set_ylabel("Y", fontproperties=font2)      #  ylabel: 设置横轴、纵轴标签及大小

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 6)
legend1 = ax2.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


# 标题
fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 14}
ax2.set_title("csdn", fontproperties=fontt, color='#000FFF',)  #   title：设置图片的标题
#===================================================================
# 画第3个图：饼图
#===================================================================
ax3=fig.add_subplot(223)
ax3.pie(x=[15,30,45,10],labels=list('ABCD'),autopct='%.0f',explode=[0,0.05,0,0])


#===================================================================
# 画第4个图：条形图
#===================================================================

ax4=fig.add_subplot(224)
ax4.bar([20,10,30,25,15],[25,15,35,30,20],color='b')



#====================================================
fontt = FontProperties(fname=fontpath+"simsun.ttf", size=12)
plt.suptitle("设置图片的标题", fontproperties=fontt, x=0.5, y=1,)
plt.subplots_adjust(left=0, bottom=0, right=1, top=0.9 , wspace=0.4, hspace=0.4)
plt.show()



#====================================================================================
#  「方式三：通过plt的subplots」subplots返回的值的类型为元组，其中包含两个元素：第一个为一个画布，第二个是子图
#====================================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,2)
# 画第1个图：折线图
x=np.arange(1,100)
ax[0][0].plot(x,x*x)
# 画第2个图：散点图
ax[0][1].scatter(np.arange(0,10), np.random.rand(10))
# 画第3个图：饼图
ax[1][0].pie(x=[15,30,45,10],labels=list('ABCD'),autopct='%.0f',explode=[0,0.05,0,0])
# 画第4个图：条形图
ax[1][1].bar([20,10,30,25,15],[25,15,35,30,20],color='b')
plt.show()

## or
fig, ax = plt.subplots(2,2)
# 画第1个图：折线图
x=np.arange(1,100)
ax[0,0].plot(x,x*x)
# 画第2个图：散点图
ax[0,1].scatter(np.arange(0,10), np.random.rand(10))
# 画第3个图：饼图
ax[1,0].pie(x=[15,30,45,10],labels=list('ABCD'),autopct='%.0f',explode=[0,0.05,0,0])
# 画第4个图：条形图
ax[1,1].bar([20,10,30,25,15],[25,15,35,30,20],color='b')
plt.show()



#====================================================================================
#  绘制不规则子图
# 前面的两个图占了221和222的位置，如果想在下面只放一个图，得把前两个当成一列，即2行1列第2个位置。
#====================================================================================



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
# 画第1个图：折线图
x=np.arange(1,100)
plt.subplot(221)
plt.plot(x,x*x)
# 画第2个图：散点图
plt.subplot(222)
plt.scatter(np.arange(0,10), np.random.rand(10))
# 画第3个图：条形图
# 前面的两个图占了221和222的位置，如果想在下面只放一个图，得把前两个当成一列，即2行1列第2个位置
plt.subplot(212)
plt.bar([20,10,30,25,15],[25,15,35,30,20],color='b')
plt.show()






#============================================= 不等间距画图 ====================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif']=['SimHei']         # 处理中文无法正常显示的问题 成功
plt.rcParams['axes.unicode_minus'] = False #负号显示


plt.xlabel("这是x轴")  # 设置x轴名称
plt.ylabel("这是y轴")  # 设置y轴名称
plt.title("这是标题")  # 设置标题


x=[1,2,3,4,5,6]                    #虚假的x值，用来等间距分割
x_index=['1','10','100','1000','10000','100000']  # x 轴显示的刻度
y=[0.1,0.15,0.2,0.3,0.35,0.5]       #y值
plt.plot(x,y,'o', marker='d')
_ = plt.xticks(x,x_index)           # 显示坐标字

plt.show()


#============================================= 不等间距画图 ====================================================

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


fig, axs = plt.subplots(1, 1, figsize=(6, 6))
y=[pow(10,i) for i in range(0,10)]
x=range(0,len(y))


axs.plot(x, y, 'r', label='BER')
axs.set_yscale('log')#设置纵坐标的缩放，写成m³格式

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font3,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('SNR', fontproperties=font)
axs.set_ylabel("log(BER)", fontproperties=font)
axs.set_title("BER", fontproperties=font)


plt.show()



#============================================= 不等间距画图 ====================================================

import numpy as np

import matplotlib.pyplot as plt

fig, ax4 = plt.subplots()
x = 10.0**np.linspace(0.0, 2.0, 15)

y = x**2.0

ax4.set_xscale("log",)

ax4.set_yscale("log",)

ax4.errorbar(x, y, xerr = 0.1 * x,yerr = 2.0 + 1.75 * y,color ="green")

ax4.set_ylim(bottom = 0.1)

fig.suptitle('matplotlib.axes.Axes.set_yscale() function Example', fontweight ="bold")

plt.show()


#============================================= 不等间距画图 ====================================================

import os
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np

BA = nx.random_graphs.barabasi_albert_graph(5000, 3)


degree = nx.degree_histogram(BA)
#生成x轴序列，从1到最大度
x = range(len(degree))
#将频次转换为频率
y = [z / float(sum(degree)) for z in degree]

#在双对数坐标轴上绘制度分布曲线
plt.loglog(x, y, 'o',  color="blue", marker='.')
#x,y后的点代表散点

plt.grid(color = 'r', linestyle = '--', linewidth = 0.5)

#显示图表
plt.show()




#============================================= 不等间距画图 ====================================================

newX = []
newY = []
for i in range(len(x)):
    if y[i] != 0 :
        newX.append(x[i])
        newY.append(y[i])

fig, ax = plt.subplots()
ax.plot(newX,newY, 'ro-')
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel('$k$', fontsize='large')
ax.set_ylabel('$p_k$', fontsize='large')
ax.legend(loc="best")
ax.set_title("degree distribution")
fig.show()








#============================================= reshape ====================================================

fig, axs = plt.subplots(1, 2, figsize=(10, 6))

axs = axs.reshape(1,2)###

t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1.1, 0.1)
s4 = np.arcsin(t1)



axs[0,0].plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)

axs[0,0].legend()
axs[0,1].plot(t, s2, color='r', linestyle='-', label='sin(x)正弦',)
axs[0,1].legend()

fig.subplots_adjust(hspace=0.8)  # 调节两个子图间的距离

# fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt)
plt.show()


#============================================= reshape ====================================================

fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# axs = axs.reshape(1,2)###

t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1.1, 0.1)
s4 = np.arcsin(t1)



axs[0].plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)

axs[0].legend()
axs[1].plot(t, s2, color='r', linestyle='-', label='sin(x)正弦',)
axs[1].legend()

fig.subplots_adjust(hspace=0.8)  # 调节两个子图间的距离

# fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt)
plt.show()



#============================================= reshape ====================================================

fig, axs = plt.subplots(2, 1, figsize=(10, 6))

axs = axs.reshape(2,1)###diff

t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1.1, 0.1)
s4 = np.arcsin(t1)



axs[0,0].plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)
axs[0,0].legend()
axs[1,0].plot(t, s2, color='r', linestyle='-', label='cos(x)正弦',)
axs[1,0].legend()

fig.subplots_adjust(hspace=0.8)  # 调节两个子图间的距离

# fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt)
plt.show()

#============================================= reshape ====================================================

fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# axs = axs.reshape(2,1)###diff

t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1.1, 0.1)
s4 = np.arcsin(t1)



axs[0].plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)

axs[1].plot(t, s2, color='r', linestyle='-', label='cos(x)正弦',)


fig.subplots_adjust(hspace=0.8)  # 调节两个子图间的距离

# fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt)
plt.show()

#================================================================================================================


X = np.arange(0, 2, 0.1)

s1 = np.sin(2*np.pi*X)
s2 = np.cos(2*np.pi*X)
s3 = np.tan(2*np.pi*X)



losslog = np.zeros((len(X),3))
losslog[:,0] = s1
losslog[:,1] = s2
losslog[:,2] = s3

loss = "MSE"
fig = plt.figure(constrained_layout=True)
for i, l in enumerate(loss):
    label = '{} Loss'.format(l)
    fig = plt.figure()

    plt.plot(X, losslog[:, i], label=label)

    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
    plt.xlabel('Epoch',fontproperties=font)
    plt.ylabel('Training loss',fontproperties=font)
    plt.title(label,fontproperties=font)
                    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
    legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    ax=plt.gca()#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

    plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(20) for label in labels] #刻度值字号


    #plt.grid(True)
    #plt.savefig(os.path.join(apath, 'TrainLossPlot_{}.pdf'.format(l['type'])))
    plt.show()
    #plt.close(fig)




#================================================================================================================

fig, axs = plt.subplots(1, 1, figsize=(10, 6))

#axs = axs.reshape(1,1)

t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1.1, 0.1)
s4 = np.arcsin(t1)



axs.plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)

axs.plot(t, s2, color='r', linestyle='-', label='cos(x)正弦',)


fig.subplots_adjust(hspace=0.8)  # 调节两个子图间的距离

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt)
plt.show()
























