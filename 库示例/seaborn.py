#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:40:24 2022

@author: jack
"""



#==============================================================================================
# https://zhuanlan.zhihu.com/p/24464836
#==============================================================================================

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt #导入

import seaborn as sns
sns.set(color_codes=True)#导入seaborn包设定颜色

np.random.seed(sum(map(ord, "distributions")))

x = np.random.normal(size=100000)
sns.distplot(x, kde=False, rug=True);#kde=False关闭核密度分布,rug表示在x轴上每个观测上生成的小细条（边际毛毯）


sns.distplot(x, bins=20, kde=False, rug=True);#设置了20个矩形条


x = np.random.gamma(6, size=20000)#生成gamma分布的数据
sns.distplot(x, kde=False, fit=stats.gamma);#fit拟合



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

# Generate a random univariate dataset
d = rs.normal(size=1000)

# Plot a simple histogram with binsize determined automatically
sns.distplot(d, kde=False, color="b", ax=axes[0, 0])

# Plot a kernel density estimate and rug plot
sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])

# Plot a filled kernel density estimate
sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

# Plot a historgram and kernel density estimate
sns.distplot(d, color="m", ax=axes[1, 1])

plt.setp(axes, yticks=[])
plt.tight_layout()


#==============================================================================================
#      https://www.jianshu.com/p/117184501316
#==============================================================================================

import numpy as np  
import matplotlib.pyplot as plt        
import seaborn as sns
import pandas as pd
sns.set()


#seaborn.displot() 用于绘制单变量分布情况
x = np.random.normal(size=1000)
sns.distplot(x)



#seaborn.jointmtplot() 用于绘制二元分布情况
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 20000)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df)


#以通过kind设置 “scatter” , “reg” , “resid” , “kde” ,“hex” 例如"hex"
sns.jointplot(x='x', y='y',data=df, kind="hex")


#seaborn.piarplot() 用于绘制数据集中特征两两之间关系
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species', size=3)


#==============================================================================================
#      http://seaborn.pydata.org/tutorial/relational.html
#==============================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips);


sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);




sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",
            data=tips);



sns.relplot(x="total_bill", y="tip", hue="smoker", style="time", data=tips);



sns.relplot(x="total_bill", y="tip", hue="size", data=tips);


sns.relplot(x="total_bill", y="tip", hue="size", palette="ch:r=-.5,l=.75", data=tips);


sns.relplot(x="total_bill", y="tip", size="size", data=tips);


sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips);


#Emphasizing continuity with line plots
df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.figure.autofmt_xdate()


df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])
sns.relplot(x="x", y="y", sort=False, kind="line", data=df);


#Aggregation and representing uncertainty¶
fmri = sns.load_dataset("fmri")
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri);


sns.relplot(x="timepoint", y="signal", ci=None, kind="line", data=fmri);

sns.relplot(x="timepoint", y="signal", kind="line", ci="sd", data=fmri);


sns.relplot(x="timepoint", y="signal", estimator=None, kind="line", data=fmri);


#Plotting subsets of data with semantic mappings
sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri);

sns.relplot(x="timepoint", y="signal", hue="region", style="event",
            kind="line", data=fmri);


sns.relplot(x="timepoint", y="signal", hue="region", style="event",
            dashes=False, markers=True, kind="line", data=fmri);



sns.relplot(x="timepoint", y="signal", hue="event", style="event",
            kind="line", data=fmri);



sns.relplot(x="timepoint", y="signal", hue="region",
            units="subject", estimator=None,
            kind="line", data=fmri.query("event == 'stim'"));




dots = sns.load_dataset("dots").query("align == 'dots'")
sns.relplot(x="time", y="firing_rate",
            hue="coherence", style="choice",
            kind="line", data=dots);



palette = sns.cubehelix_palette(light=.8, n_colors=6)
sns.relplot(x="time", y="firing_rate",
            hue="coherence", style="choice",
            palette=palette,
            kind="line", data=dots);




from matplotlib.colors import LogNorm
palette = sns.cubehelix_palette(light=.7, n_colors=6)
sns.relplot(x="time", y="firing_rate",
            hue="coherence", style="choice",
            hue_norm=LogNorm(),
            kind="line",
            data=dots.query("coherence > 0"));



sns.relplot(x="time", y="firing_rate",
            size="coherence", style="choice",
            kind="line", data=dots);


sns.relplot(x="time", y="firing_rate",
           hue="coherence", size="choice",
           palette=palette,
           kind="line", data=dots);


#Plotting with date data
df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.figure.autofmt_xdate()


sns.relplot(x="total_bill", y="tip", hue="smoker",
            col="time", data=tips);



sns.relplot(x="timepoint", y="signal", hue="subject",
            col="region", row="event", height=3,
            kind="line", estimator=None, data=fmri);





sns.relplot(x="timepoint", y="signal", hue="event", style="event",
            col="subject", col_wrap=5,
            height=3, aspect=.75, linewidth=2.5,
            kind="line", data=fmri.query("region == 'frontal'"));








#==============================================================================================
#      http://seaborn.pydata.org/tutorial/distributions.html
#==============================================================================================



#Plotting univariate histograms
penguins = sns.load_dataset("penguins")
sns.displot(penguins, x="flipper_length_mm")

#Choosing the bin size
sns.displot(penguins, x="flipper_length_mm", binwidth=3)


sns.displot(penguins, x="flipper_length_mm", bins=20)



tips = sns.load_dataset("tips")
sns.displot(tips, x="size")


sns.displot(tips, x="size", bins=[1, 2, 3, 4, 5, 6, 7])


sns.displot(tips, x="size", discrete=True)


sns.displot(tips, x="day", shrink=.8)




#Conditioning on other variables
sns.displot(penguins, x="flipper_length_mm", hue="species")

sns.displot(penguins, x="flipper_length_mm", hue="species", element="step")

sns.displot(penguins, x="flipper_length_mm", hue="species", multiple="stack")

sns.displot(penguins, x="flipper_length_mm", hue="sex", multiple="dodge")


sns.displot(penguins, x="flipper_length_mm", col="sex")



#Normalized histogram statistics

sns.displot(penguins, x="flipper_length_mm", hue="species", stat="density")


sns.displot(penguins, x="flipper_length_mm", hue="species", stat="density", common_norm=False)




sns.displot(penguins, x="flipper_length_mm", hue="species", stat="probability")




#Kernel density estimation
sns.displot(penguins, x="flipper_length_mm", kind="kde")

sns.displot(penguins, x="flipper_length_mm", kind="kde", bw_adjust=.25)


sns.displot(penguins, x="flipper_length_mm", kind="kde", bw_adjust=2)



#Conditioning on other variables
sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde")

sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", multiple="stack")

sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", fill=True)





#Kernel density estimation pitfalls
sns.displot(tips, x="total_bill", kind="kde")

sns.displot(tips, x="total_bill", kind="kde", cut=0)



diamonds = sns.load_dataset("diamonds")
sns.displot(diamonds, x="carat", kind="kde")

sns.displot(diamonds, x="carat")



sns.displot(diamonds, x="carat", kde=True)





#Empirical cumulative distributions

sns.displot(penguins, x="flipper_length_mm", kind="ecdf")

sns.displot(penguins, x="flipper_length_mm", hue="species", kind="ecdf")


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm")


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde")



sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")




sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", kind="kde")

sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", binwidth=(2, .5))


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", binwidth=(2, .5), cbar=True)




sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde", thresh=.2, levels=4)


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde", levels=[.01, .05, .1, .8])


sns.displot(diamonds, x="price", y="clarity", log_scale=(True, False))



sns.displot(diamonds, x="color", y="clarity")


#Plotting joint and marginal distributions

sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")

sns.jointplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="species",
    kind="kde"
)



g = sns.JointGrid(data=penguins, x="bill_length_mm", y="bill_depth_mm")
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot)

sns.displot(
    penguins, x="bill_length_mm", y="bill_depth_mm",
    kind="kde", rug=True
)



sns.relplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
sns.rugplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")


sns.pairplot(penguins)




g = sns.PairGrid(penguins)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)






#==============================================================================================
#    http://seaborn.pydata.org/tutorial/categorical.html
#==============================================================================================
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)


tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", data=tips)



sns.catplot(x="day", y="total_bill", jitter=False, data=tips)




sns.catplot(x="day", y="total_bill", kind="swarm", data=tips)




sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips)






sns.catplot(x="size", y="total_bill", data=tips)





sns.catplot(x="smoker", y="tip", order=["No", "Yes"], data=tips)


sns.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=tips)




#Distributions of observations within categories
sns.catplot(x="day", y="total_bill", kind="box", data=tips)

sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips)


tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
sns.catplot(x="day", y="total_bill", hue="weekend",
            kind="box", dodge=False, data=tips)


diamonds = sns.load_dataset("diamonds")
sns.catplot(x="color", y="price", kind="boxen",
            data=diamonds.sort_values("color"))





#Violinplots
sns.catplot(x="total_bill", y="day", hue="sex",
            kind="violin", data=tips)



sns.catplot(x="total_bill", y="day", hue="sex",
            kind="violin", bw=.15, cut=0,
            data=tips)




sns.catplot(x="day", y="total_bill", hue="sex",
            kind="violin", split=True, data=tips)



sns.catplot(x="day", y="total_bill", hue="sex",
            kind="violin", inner="stick", split=True,
            palette="pastel", data=tips)

g = sns.catplot(x="day", y="total_bill", kind="violin", inner=None, data=tips)
sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax)



titanic = sns.load_dataset("titanic")
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)




sns.catplot(x="deck", kind="count", palette="ch:.25", data=titanic)


sns.catplot(y="deck", hue="class", kind="count",
            palette="pastel", edgecolor=".6",
            data=titanic)



#Point plots
sns.catplot(x="sex", y="survived", hue="class", kind="point", data=titanic)





sns.catplot(x="class", y="survived", hue="sex",
            palette={"male": "g", "female": "m"},
            markers=["^", "o"], linestyles=["-", "--"],
            kind="point", data=titanic)









#Plotting “wide-form” data

iris = sns.load_dataset("iris")
sns.catplot(data=iris, orient="h", kind="box")



sns.violinplot(x=iris.species, y=iris.sepal_length)

f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="deck", data=titanic, color="c")


sns.catplot(x="day", y="total_bill", hue="smoker",
            col="time", aspect=.7,
            kind="swarm", data=tips)




g = sns.catplot(x="fare", y="survived", row="class",
                kind="box", orient="h", height=1.5, aspect=4,
                data=titanic.query("fare > 0"))
g.set(xscale="log")





#==============================================================================================
#    http://seaborn.pydata.org/tutorial/aesthetics.html
#==============================================================================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
sinplot()





sns.set_theme()
sinplot()





#Seaborn figure styles

sns.set_style("whitegrid")
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data);



sns.set_style("dark")
sinplot()

sns.set_style("white")
sinplot()



sns.set_style("ticks")
sinplot()




#Removing axes spines

sinplot()
sns.despine()



f, ax = plt.subplots()
sns.violinplot(data=data)
sns.despine(offset=10, trim=True);



sns.set_style("whitegrid")
sns.boxplot(data=data, palette="deep")
sns.despine(left=True)


#Temporarily setting figure style

f = plt.figure(figsize=(6, 6))
gs = f.add_gridspec(2, 2)

with sns.axes_style("darkgrid"):
    ax = f.add_subplot(gs[0, 0])
    sinplot()

with sns.axes_style("white"):
    ax = f.add_subplot(gs[0, 1])
    sinplot()

with sns.axes_style("ticks"):
    ax = f.add_subplot(gs[1, 0])
    sinplot()

with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[1, 1])
    sinplot()

f.tight_layout()




#Overriding elements of the seaborn styles
sns.axes_style()


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sinplot()




#Scaling plot elements

sns.set_theme()




sns.set_context("paper")
sinplot()

sns.set_context("talk")
sinplot()

sns.set_context("poster")
sinplot()



sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sinplot()









#==============================================================================================
#    http://seaborn.pydata.org/tutorial/axis_grids.html
#==============================================================================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time")



g = sns.FacetGrid(tips, col="time")
g.map(sns.histplot, "tip")



g = sns.FacetGrid(tips, col="sex", hue="smoker")
g.map(sns.scatterplot, "total_bill", "tip", alpha=.7)
g.add_legend()


g = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
g.map(sns.regplot, "size", "total_bill", color=".3", fit_reg=False, x_jitter=.1)



g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
g.map(sns.barplot, "sex", "total_bill", order=["Male", "Female"])





ordered_days = tips.day.value_counts().index
g = sns.FacetGrid(tips, row="day", row_order=ordered_days,
                  height=1.7, aspect=4,)
g.map(sns.kdeplot, "total_bill")





pal = dict(Lunch="seagreen", Dinner=".7")
g = sns.FacetGrid(tips, hue="time", palette=pal, height=5)
g.map(sns.scatterplot, "total_bill", "tip", s=100, alpha=.5)
g.add_legend()


attend = sns.load_dataset("attention").query("subject <= 12")
g = sns.FacetGrid(attend, col="subject", col_wrap=4, height=2, ylim=(0, 10))
g.map(sns.pointplot, "solutions", "score", order=[1, 2, 3], color=".3", ci=None)


with sns.axes_style("white"):
    g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, height=2.5)
g.map(sns.scatterplot, "total_bill", "tip", color="#334488")
g.set_axis_labels("Total bill (US Dollars)", "Tip")
g.set(xticks=[10, 30, 50], yticks=[2, 6, 10])
g.figure.subplots_adjust(wspace=.02, hspace=.02)








g = sns.FacetGrid(tips, col="smoker", margin_titles=True, height=4)
g.map(plt.scatter, "total_bill", "tip", color="#338844", edgecolor="white", s=50, lw=1)
for ax in g.axes_dict.values():
    ax.axline((0, 0), slope=.2, c=".2", ls="--", zorder=0)
g.set(xlim=(0, 60), ylim=(0, 14))






#Using custom functions

from scipy import stats
def quantile_plot(x, **kwargs):
    quantiles, xr = stats.probplot(x, fit=False)
    plt.scatter(xr, quantiles, **kwargs)

g = sns.FacetGrid(tips, col="sex", height=4)
g.map(quantile_plot, "total_bill")




def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)

g = sns.FacetGrid(tips, col="smoker", height=4)
g.map(qqplot, "total_bill", "tip")



g = sns.FacetGrid(tips, hue="time", col="sex", height=4)
g.map(qqplot, "total_bill", "tip")
g.add_legend()






def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

with sns.axes_style("dark"):
    g = sns.FacetGrid(tips, hue="time", col="time", height=4)
g.map(hexbin, "total_bill", "tip", extent=[0, 50, 0, 10]);




#Plotting pairwise data relationships

iris = sns.load_dataset("iris")
g = sns.PairGrid(iris)
g.map(sns.scatterplot)



g = sns.PairGrid(iris)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)




g = sns.PairGrid(iris, hue="species")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()






g = sns.PairGrid(iris, vars=["sepal_length", "sepal_width"], hue="species")
g.map(sns.scatterplot)



g = sns.PairGrid(iris)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False)





g = sns.PairGrid(tips, y_vars=["tip"], x_vars=["total_bill", "size"], height=4)
g.map(sns.regplot, color=".3")
g.set(ylim=(-1, 11), yticks=[0, 5, 10])




g = sns.PairGrid(tips, hue="size", palette="GnBu_d")
g.map(plt.scatter, s=50, edgecolor="white")
g.add_legend()



sns.pairplot(iris, hue="species", height=2.5)



g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", height=2.5)


print("=="*40)
# https://www.heywhale.com/mw/project/5e0596d72823a10036b0385c
"""
Matplotlib试着让简单的事情更加简单，困难的事情变得可能，而Seaborn就是让困难的东西更加简单。

seaborn是针对统计绘图的，一般来说，seaborn能满足数据分析90%的绘图需求。

Seaborn其实是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，在大多数情况下使用seaborn就能做出很具有吸引力的图，应该把Seaborn视为matplotlib的补充，而不是替代物。

用matplotlib最大的困难是其默认的各种参数，而Seaborn则完全避免了这一问题。

seaborn一共有5个大类21种图，分别是：
Relational plots 关系类图表

relplot() 关系类图表的接口，其实是下面两种图的集成，通过指定kind参数可以画出下面的两种图
scatterplot() 散点图
lineplot() 折线图
Categorical plots 分类图表

catplot() 分类图表的接口，其实是下面八种图表的集成，，通过指定kind参数可以画出下面的八种图
stripplot() 分类散点图
swarmplot() 能够显示分布密度的分类散点图
boxplot() 箱图
violinplot() 小提琴图
boxenplot() 增强箱图
pointplot() 点图
barplot() 条形图
countplot() 计数图
Distribution plot 分布图

jointplot() 双变量关系图
pairplot() 变量关系组图
distplot() 直方图，质量估计图
kdeplot() 核函数密度估计图
rugplot() 将数组中的数据点绘制为轴上的数据
Regression plots 回归图

lmplot() 回归模型图
regplot() 线性回归图
residplot() 线性回归残差图
Matrix plots 矩阵图

heatmap() 热力图
clustermap() 聚集图
"""


# 如果不添加这句，是无法直接在jupyter里看到图的
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

























print("=="*40)
# https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/716978/

import seaborn as sns

sns.set(font_scale=1.5,style="white")



#本次試用的數據集是Seaborn內置的tips小費數據集：
data=sns.load_dataset("tips")
data.head(5) 




#我們先來看一下lmplot是什麼樣的
sns.lmplot(x="total_bill",y="tip",data=data)




#可以看到lmplot對所選數據集進行了一元線性迴歸，擬合出了一條最佳的直線，
#接下來進入具體參數的演示。
#col:根據所指定屬性在列上分類
#row:根據所指定屬性在行上分類
sns.lmplot(x="total_bill",y="tip",data=data,row="sex",col="smoker")





#結合我們的數據集，看上圖的橫縱座標就可以明白這兩個參數的用法

#col_wrap:指定每行的列數，最多等於col參數所對應的不同類別的數量

sns.lmplot(x="total_bill",y="tip",data=data,col="day",col_wrap=4) 

sns.lmplot(x="total_bill",y="tip",data=data,col="day",col_wrap=2) 



#aspect:控制圖的長寬比
sns.lmplot(x="total_bill",y="tip",data=data,aspect=1)  
#長度比寬度等於一比一，即正方形



sns.lmplot(x="total_bill",y="tip",data=data,aspect=1.5)  
#長度比寬度等於1:1.5，可以看到橫軸更長一點






#sharex:共享x軸刻度（默認為True）
#sharey:共享y軸刻度（默認為True）
sns.lmplot(x="total_bill",y="tip",data=data,row="sex",col="smoker",sharex=False)
#可以看到設置為False時，各個子圖的x軸的5#座標刻度是不一樣的



#hue:用於分類
sns.lmplot(x="total_bill",y="tip",data=data,hue="sex",palette="husl") 



#ci:控制迴歸的置信區間（有學過統計學的同學們應該都是知道滴）
#顯示0.95的置信區間
sns.lmplot(x="total_bill",y="tip",data=data,ci=95)






#x_jitter:給x軸隨機增加噪音點
#y_jitter:給y軸隨機增加噪音點
#設置這兩個參數不影響最後的迴歸直線
sns.lmplot(x="size",y="tip",data=data,x_jitter=False) 





sns.lmplot(x="size",y="tip",data=data,x_jitter=True)    
#可以看到剛才的一列一列的數據點被隨機   
#打亂了，但不會影響到最後的迴歸直線





#order:控制進行迴歸的冪次（一次以上即是多項式迴歸）

sns.lmplot(x="total_bill",y="tip",data=data,order=1)  #一元線性迴歸
sns.lmplot(x="total_bill",y="tip",data=data,order=2) #次數最高為2
sns.lmplot(x="total_bill",y="tip",data=data,order=3) #次數最高為3


print("=="*40)















































































































































































