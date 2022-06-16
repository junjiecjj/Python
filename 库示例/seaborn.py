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
print(" https://www.heywhale.com/mw/project/5e0596d72823a10036b0385c   important ")
print("=="*40)

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


#有一套的参数可以控制绘图元素的比例。
#首先，让我们通过set()重置默认的参数：
#有五种seaborn的风格，它们分别是：darkgrid, whitegrid, dark, white, ticks。它们各自适合不同的应用和个人喜好。默认的主题是darkgrid。
sns.set(style="ticks")

# Load the example dataset for Anscombe's quartet
df = sns.load_dataset("anscombe")
df.head()

# lmplot是用来绘制回归图的，通过lmplot我们可以直观地总览数据的内在关系。
# Show the results of a linear regression within each dataset
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})



sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})




sns.set()

# Load the iris dataset
iris = sns.load_dataset("iris")

# Plot sepal with as a function of sepal_length across days
g = sns.lmplot(x="sepal_length", y="sepal_width", hue="species",
               truncate=True, height=5, data=iris)

# Use more informative axis labels than are provided by default
g.set_axis_labels("Sepal length (mm)", "Sepal width (mm)")





sns.set(style="darkgrid")

# Load the example titanic dataset
df = sns.load_dataset("titanic")

# Make a custom palette with gendered colors
pal = dict(male="#6495ED", female="#F08080")

# Show the survival proability as a function of age and sex
g = sns.lmplot(x="age", y="survived", col="sex", hue="sex", data=df,
               palette=pal, y_jitter=.02, logistic=True)
g.set(xlim=(0, 80), ylim=(-.05, 1.05))






#kdeplot(核密度估计图)
#核密度估计(kernel density estimation)是在概率论中用来估计未知的密度函数，属于非参数检验方法之一。通过核密度估计图可以比较直观的看出数据样本本身的分布特征。具体用法如下：
sns.set(style="dark")
rs = np.random.RandomState(50)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# Rotate the starting point around the cubehelix hue circle
for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    x, y = rs.randn(2, 50)
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=ax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

f.tight_layout()

#==========================================================
sns.set(style="darkgrid")
iris = sns.load_dataset("iris")

# Subset the iris dataset by species
setosa = iris.query("species == 'setosa'")
virginica = iris.query("species == 'virginica'")

# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "virginica", size=16, color=blue)
ax.text(3.8, 4.5, "setosa", size=16, color=red)



"""
FacetGrid
是一个绘制多个图表（以网格形式显示）的接口。

步骤：

1、实例化对象

2、map，映射到具体的 seaborn 图表类型

3、添加图例
"""
"""
Overlapping densities ('ridge plot')
====================================
"""
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create the data
rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list("ABCDEFGHIJ"), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)
df["x"] += m

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "x")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)


"""
FacetGrid with custom projection
================================

"""
sns.set()

# Generate an example radial datast
r = np.linspace(0, 10, num=100)
df = pd.DataFrame({'r': r, 'slow': r, 'medium': 2 * r, 'fast': 4 * r})

# Convert the dataframe to long-form or "tidy" format
df = pd.melt(df, id_vars=['r'], var_name='speed', value_name='theta')

# Set up a grid of axes with a polar projection
g = sns.FacetGrid(df, col="speed", hue="speed",
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)

# Draw a scatterplot onto each axes in the grid
g.map(sns.scatterplot, "theta", "r")




"""
Facetting histograms by subsets of data
=======================================

"""
sns.set(style="darkgrid")

tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "total_bill", color="steelblue", bins=bins)






"""
Plotting on a large number of facets
====================================

"""
sns.set(style="ticks")

# Create a dataset with many short random walks
rs = np.random.RandomState(4)
pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
pos -= pos[:, 0, np.newaxis]
step = np.tile(range(5), 20)
walk = np.repeat(range(20), 5)
df = pd.DataFrame(np.c_[pos.flat, step, walk],
                  columns=["position", "step", "walk"])

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(df, col="walk", hue="walk", palette="tab20c",
                     col_wrap=4, height=1.5)

# Draw a horizontal line to show the starting point
grid.map(plt.axhline, y=0, ls=":", c=".5")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "step", "position", marker="o")

# Adjust the tick positions and labels
grid.set(xticks=np.arange(5), yticks=[-3, 3],
         xlim=(-.5, 4.5), ylim=(-3.5, 3.5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)




#distplot(单变量分布直方图)
#在seaborn中想要对单变量分布进行快速了解最方便的就是使用distplot()函数，默认情况下它将绘制一个直方图，并且可以同时画出核密度估计(KDE)。

"""
Distribution plot options
=========================

"""
sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

# Generate a random univariate dataset
d = rs.normal(size=100)

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


"""
lineplot
绘制线段

seaborn里的lineplot函数所传数据必须为一个pandas数组，这一点跟matplotlib里有较大区别，并且一开始使用较为复杂，sns.lineplot里有几个参数值得注意。

x: plot图的x轴label

y: plot图的y轴label

ci: 与估计器聚合时绘制的置信区间的大小

data: 所传入的pandas数组
"""
"""
Timeseries plot with error bands
================================

"""
sns.set(style="darkgrid")

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")

# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri)



"""
Lineplot from a wide-form dataset
=================================

"""

sns.set(style="whitegrid")

rs = np.random.RandomState(365)
values = rs.randn(365, 4).cumsum(axis=0)
dates = pd.date_range("1 1 2016", periods=365, freq="D")
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
data = data.rolling(7).mean()

sns.lineplot(data=data, palette="tab10", linewidth=2.5)


#relplot
#这是一个图形级别的函数，它用散点图和线图两种常用的手段来表现统计关系。
"""
Line plots on multiple facets
=============================

"""

sns.set(style="ticks")

dots = sns.load_dataset("dots")

# Define a palette to ensure that colors will be
# shared across the facets
palette = dict(zip(dots.coherence.unique(),
                   sns.color_palette("rocket_r", 6)))

# Plot the lines on two facets
sns.relplot(x="time", y="firing_rate",
            hue="coherence", size="choice", col="align",
            size_order=["T1", "T2"], palette=palette,
            height=5, aspect=.75, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=dots)


# boxplot
# 箱形图（Box-plot）又称为盒须图、盒式图或箱线图，是一种用作显示一组数据分散情况资料的统计图。它能显示出一组数据的最大值、最小值、中位数及上下四分位数。
"""
Grouped boxplots
================
"""
sns.set(style="ticks", palette="pastel")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="day", y="total_bill",
            hue="smoker", palette=["m", "g"],
            data=tips)
sns.despine(offset=10, trim=True)






# violinplot
# violinplot与boxplot扮演类似的角色，它显示了定量数据在一个（或多个）分类变量的多个层次上的分布，这些分布可以进行比较。不像箱形图中所有绘图组件都对应于实际数据点，小提琴绘图以基础分布的核密度估计为特征。
"""
Violinplots with observations
=============================

"""


sns.set()

# Create a random dataset across several variables
rs = np.random.RandomState(0)
n, p = 40, 8
d = rs.normal(0, 2, (n, p))
d += np.log(np.arange(1, p + 1)) * -5 + 10

# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(p, rot=-.5, dark=.3)

# Show each distribution with both violins and points
sns.violinplot(data=d, palette=pal, inner="points")




"""
Grouped violinplots with split violins
======================================
"""
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="day", y="total_bill", hue="smoker",
               split=True, inner="quart",
               palette={"Yes": "y", "No": "b"},
               data=tips)
sns.despine(left=True)




"""
Violinplot from a wide-form dataset
===================================

"""

sns.set(style="whitegrid")

# Load the example dataset of brain network correlations
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Pull out a specific subset of networks
used_networks = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# Compute the correlation matrix and average over networks
corr_df = df.corr().groupby(level="network").mean()
corr_df.index = corr_df.index.astype(int)
corr_df = corr_df.sort_index().T

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 6))

# Draw a violinplot with a narrower bandwidth than the default
sns.violinplot(data=corr_df, palette="Set3", bw=.2, cut=1, linewidth=1)

# Finalize the figure
ax.set(ylim=(-.7, 1.05))
sns.despine(left=True, bottom=True)



#heatmap热力图
#利用热力图可以看数据表里多个特征两两的相似度。



"""
Annotated heatmaps
==================

"""
sns.set()

# Load the example flights dataset and conver to long-form
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)

"""
Plotting a diagonal correlation matrix
======================================

"""
from string import ascii_letters

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})






#jointplot
#用于2个变量的画图

"""
Joint kernel density estimate
=============================
"""
sns.set(style="white")

# Generate a random correlated bivariate dataset
rs = np.random.RandomState(5)
mean = [0, 0]
cov = [(1, .5), (.5, 1)]
x1, x2 = rs.multivariate_normal(mean, cov, 500).T
x1 = pd.Series(x1, name="$X_1$")
x2 = pd.Series(x2, name="$X_2$")

# Show the joint distribution using kernel density estimation
g = sns.jointplot(x1, x2, kind="kde", height=7, space=0)



#HexBin图

#直方图的双变量类似物被称为“hexbin”图，因为它显示了落在六边形仓内的观测数。该图适用于较大的数据集。
"""
Hexbin plot with marginal distributions
=======================================
"""
sns.set(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

sns.jointplot(x, y, kind="hex", color="#4CB391")


"""
Linear regression with marginal distributions
=============================================

"""

sns.set(style="darkgrid")

tips = sns.load_dataset("tips")
g = sns.jointplot("total_bill", "tip", data=tips, kind="reg",
                  xlim=(0, 60), ylim=(0, 12), color="m", height=7)



#barplot(条形图)
#条形图表示数值变量与每个矩形高度的中心趋势的估计值，并使用误差线提供关于该估计值附近的不确定性的一些指示。
"""
Horizontal bar plots
====================
"""
sns.set(style="whitegrid")

# Load the example car crash dataset
crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))
# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="total", y="abbrev", data=crashes,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="alcohol", y="abbrev", data=crashes,
            label="Alcohol-involved", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)





"""
catplot
分类图表的接口，通过指定kind参数可以画出下面的八种图

stripplot() 分类散点图

swarmplot() 能够显示分布密度的分类散点图

boxplot() 箱图

violinplot() 小提琴图

boxenplot() 增强箱图

pointplot() 点图

barplot() 条形图

countplot() 计数图
"""


"""
Grouped barplots
================
"""
sns.set(style="whitegrid")

# Load the example Titanic dataset
titanic = sns.load_dataset("titanic")

# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="class", y="survived", hue="sex", data=titanic,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")




"""
Plotting a three-way ANOVA
==========================

"""

sns.set(style="whitegrid")

# Load the example exercise dataset
df = sns.load_dataset("exercise")

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(x="time", y="pulse", hue="kind", col="diet",
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=df)
g.despine(left=True)


#pointplot
#点图代表散点图位置的数值变量的中心趋势估计，并使用误差线提供关于该估计的不确定性的一些指示。点图可能比条形图更有用于聚焦一个或多个分类变量的不同级别之间的比较。他们尤其善于表现交互作用：一个分类变量的层次之间的关系如何在第二个分类变量的层次之间变化。连接来自相同色调等级的每个点的线允许交互作用通过斜率的差异进行判断，这比对几组点或条的高度比较容易。


"""
Conditional means with observations
===================================

"""
sns.set(style="whitegrid")
iris = sns.load_dataset("iris")

# "Melt" the dataset to "long-form" or "tidy" representation
iris = pd.melt(iris, "species", var_name="measurement")

# Initialize the figure
f, ax = plt.subplots()
sns.despine(bottom=True, left=True)

# Show each observation with a scatterplot
sns.stripplot(x="value", y="measurement", hue="species",
              data=iris, dodge=True, jitter=True,
              alpha=.25, zorder=1)

# Show the conditional means
sns.pointplot(x="value", y="measurement", hue="species",
              data=iris, dodge=.532, join=False, palette="dark",
              markers="d", scale=.75, ci=None)

# Improve the legend 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[3:], labels[3:], title="species",
          handletextpad=0, columnspacing=1,
          loc="lower right", ncol=3, frameon=True)



#scatterplot(散点图)¶
"""
Scatterplot with categorical and numerical semantics
====================================================
"""
sns.set(style="whitegrid")

# Load the example iris dataset
diamonds = sns.load_dataset("diamonds")

# Draw a scatter plot while assigning point colors and sizes to different
# variables in the dataset
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
sns.scatterplot(x="carat", y="price",
                hue="clarity", size="depth",
                palette="ch:r=-.2,d=.3_r",
                hue_order=clarity_ranking,
                sizes=(1, 8), linewidth=0,
                data=diamonds, ax=ax)


#boxenplot（增强箱图）¶
"""
Plotting large distributions
============================

"""
sns.set(style="whitegrid")

diamonds = sns.load_dataset("diamonds")
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

sns.boxenplot(x="clarity", y="carat",
              color="b", order=clarity_ranking,
              scale="linear", data=diamonds)



#Scatterplot（散点图）¶
"""
Scatterplot with continuous hues and sizes
==========================================

"""

sns.set()

# Load the example iris dataset
planets = sns.load_dataset("planets")

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
ax = sns.scatterplot(x="distance", y="orbital_period",
                     hue="year", size="mass",
                     palette=cmap, sizes=(10, 200),
                     data=planets)




"""
Scatterplot with marginal ticks
===============================
"""
sns.set(style="white", color_codes=True)

# Generate a random bivariate dataset
rs = np.random.RandomState(9)
mean = [0, 0]
cov = [(1, 0), (0, 2)]
x, y = rs.multivariate_normal(mean, cov, 100).T

# Use JointGrid directly to draw a custom plot
grid = sns.JointGrid(x, y, space=0, height=6, ratio=50)
grid.plot_joint(plt.scatter, color="g")
grid.plot_marginals(sns.rugplot, height=1, color="g")





# PairGrid
# 用于绘制数据集中成对关系的子图网格。
"""
Paired density and scatterplot matrix
=====================================

"""

sns.set(style="white")

df = sns.load_dataset("iris")

g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot)
g.map_diag(sns.kdeplot, lw=3)



"""
Paired categorical plots
========================

"""
sns.set(style="whitegrid")

# Load the example Titanic dataset
titanic = sns.load_dataset("titanic")

# Set up a grid to plot survival probability against several variables
g = sns.PairGrid(titanic, y_vars="survived",
                 x_vars=["class", "sex", "who", "alone"],
                 height=5, aspect=.5)

# Draw a seaborn pointplot onto each Axes
g.map(sns.pointplot, scale=1.3, errwidth=4, color="xkcd:plum")
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)



#residplot
#线性回归残差图

"""
Plotting model residuals
========================

"""

sns.set(style="whitegrid")

# Make an example dataset with y ~ x
rs = np.random.RandomState(7)
x = rs.normal(2, 1, 75)
y = 2 + 1.5 * x + rs.normal(0, 2, 75)

# Plot the residuals after fitting a linear model
sns.residplot(x, y, lowess=True, color="g")

"""
Scatterplot with varying point sizes and hues
==============================================

"""
sns.set(style="white")

# Load the example mpg dataset
mpg = sns.load_dataset("mpg")

# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="horsepower", y="mpg", hue="origin", size="weight",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=mpg)






#swarmplot
#能够显示分布密度的分类散点图
"""
Scatterplot with categorical variables
======================================

"""

sns.set(style="whitegrid", palette="muted")

# Load the example iris dataset
iris = sns.load_dataset("iris")

# "Melt" the dataset to "long-form" or "tidy" representation
iris = pd.melt(iris, "species", var_name="measurement")

# Draw a categorical scatterplot to show each observation
sns.swarmplot(x="measurement", y="value", hue="species",
              palette=["r", "c", "y"], data=iris)



#pairplot
#变量关系组图
"""
Scatterplot Matrix
==================

"""

sns.set(style="ticks")

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")



#clustermap
#聚集图
"""
Discovering structure in heatmap data
=====================================

"""

sns.set()

# Load the brain networks example dataset
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Select a subset of the networks
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# Create a categorical palette to identify the networks
network_pal = sns.husl_palette(8, s=.45)
network_lut = dict(zip(map(str, used_networks), network_pal))

# Convert the palette to vectors that will be drawn on the side of the matrix
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=df.columns).map(network_lut)

# Draw the full plot
sns.clustermap(df.corr(), center=0, cmap="vlag",
               row_colors=network_colors, col_colors=network_colors,
               linewidths=.75, figsize=(13, 13))





print("=="*40)
print(" https://huhuhang.com/post/machine-learning/seaborn-basic    important")
print("=="*40)


import matplotlib.pyplot as plt


x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
y_bar = [3, 4, 6, 8, 9, 10, 9, 11, 7, 8]
y_line = [2, 3, 5, 7, 8, 9, 8, 10, 6, 7]

plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')


import seaborn as sns

sns.set()  # 声明使用 Seaborn 样式

plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')



"""
我们可以发现，相比于 Matplotlib 默认的纯白色背景，Seaborn 默认的浅灰色网格背景看起来的确要细腻舒适一些。而柱状图的色调、坐标轴的字体大小也都有一些变化。
sns.set() 的默认参数为：
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
其中：
context='' 参数控制着默认的画幅大小，分别有 {paper, notebook, talk, poster} 四个值。其中，poster > talk > notebook > paper。
style='' 参数控制默认样式，分别有 {darkgrid, whitegrid, dark, white, ticks}，你可以自行更改查看它们之间的不同。
palette='' 参数为预设的调色板。分别有 {deep, muted, bright, pastel, dark, colorblind} 等，你可以自行更改查看它们之间的不同。
剩下的 font='' 用于设置字体，font_scale= 设置字体大小，color_codes= 不使用调色板而采用先前的 'r' 等色彩缩写。


"""

"""
Seaborn 绘图 API
Seaborn 一共拥有 50 多个 API 类，相比于 Matplotlib 数千个的规模，可以算作是短小精悍了。其中，根据图形的适应场景，Seaborn 的绘图方法大致分类 6 类，分别是：关联图、类别图、分布图、回归图、矩阵图和组合图。而这 6 大类下面又包含不同数量的绘图函数。
接下来，我们就通过实际数据进行演示，使用 Seaborn 绘制不同适应场景的图形。
关联图
当我们需要对数据进行关联性分析时，可能会用到 Seaborn 提供的以下几个 API。
关联性分析	介绍
relplot	绘制关系图
scatterplot	多维度分析散点图
lineplot	多维度分析线形图
relplot 是 relational plots 的缩写，其可以用于呈现数据之后的关系，主要有散点图和条形图 2 种样式。我们载入鸢尾花示例数据集。
在绘图之前，先熟悉一下 iris 鸢尾花数据集。数据集总共 150 行，由 5 列组成。分别代表：萼片长度、萼片宽度、花瓣长度、花瓣宽度、花的类别。其中，前四列均为数值型数据，最后一列花的分类为三种，分别是：Iris Setosa、Iris Versicolour、Iris Virginica。

"""
iris = sns.load_dataset("iris")
print(f"iris.head() = {iris.head()}")



#此时，我们指定 $x$ 和 $y$ 的特征，默认可以绘制出散点图。
sns.relplot(x="sepal_length", y="sepal_width", data=iris)




#但是，上图并不能看出数据类别之间的联系，如果我们加入类别特征对数据进行着色，就更加直观了。
sns.relplot(x="sepal_length", y="sepal_width", hue="species", data=iris)





#Seaborn 的函数都有大量实用的参数，例如我们指定 style 参数可以赋予不同类别的散点不同的形状。更多的参数，希望大家通过阅读官方文档了解。
sns.relplot(x="sepal_length", y="sepal_width",
            hue="species", style="species", data=iris)

#不只是散点图，该方法还支持线形图，只需要指定 kind="line" 参数即可。线形图和散点图适用于不同类型的数据。线形态绘制时还会自动给出 95% 的置信区间。
sns.relplot(x="sepal_length", y="petal_length",
            hue="species", style="species", kind="line", data=iris)

"""
你会发现，上面我们一个提到了 3 个 API，分别是：relplot，scatterplot 和 lineplot。实际上，你可以把我们已经练习过的 relplot 看作是 scatterplot 和 lineplot 的结合版本。'
"""


#例如上方的图，我们也可以使用 lineplot 函数绘制，你只需要取消掉 relplot 中的 kind 参数即可。
sns.lineplot(x="sepal_length", y="petal_length",
             hue="species", style="species", data=iris)


"""
类别图
与关联图相似，类别图的 Figure-level 接口是 catplot，其为 categorical plots 的缩写。而 catplot 实际上是如下 Axes-level 绘图 API 的集合：
分类散点图：
stripplot() (kind="strip")
swarmplot() (kind="swarm")
分类分布图：
boxplot() (kind="box")
violinplot() (kind="violin")
boxenplot() (kind="boxen")
分类估计图：
pointplot() (kind="point")
barplot() (kind="bar")
countplot() (kind="count")
下面，我们看一下 catplot 绘图效果。该方法默认是绘制 kind="strip" 散点图。
"""
sns.catplot(x="sepal_length", y="species", data=iris)


#kind="swarm" 可以让散点按照 beeswarm 的方式防止重叠，可以更好地观测数据分布。
sns.catplot(x="sepal_length", y="species", kind="swarm", data=iris)





#同理，hue= 参数可以给图像引入另一个维度，由于 iris 数据集只有一个类别列，我们这里就不再添加 hue= 参数了。如果一个数据集有多个类别，hue= 参数就可以让数据点有更好的区分。
#接下来，我们依次尝试其他几种图形的绘制效果。绘制箱线图：
sns.catplot(x="sepal_length", y="species", kind="box", data=iris)


#绘制小提琴图：
sns.catplot(x="sepal_length", y="species", kind="violin", data=iris)


#绘制增强箱线图：
sns.catplot(x="species", y="sepal_length", kind="boxen", data=iris)

#绘制点线图：
sns.catplot(x="sepal_length", y="species", kind="point", data=iris)

#绘制条形图：
sns.catplot(x="sepal_length", y="species", kind="bar", data=iris)

#绘制计数条形图：
sns.catplot(x="species", kind="count", data=iris)


"""
分布图
分布图主要是用于可视化变量的分布情况，一般分为单变量分布和多变量分布。当然这里的多变量多指二元变量，更多的变量无法绘制出直观的可视化图形。
Seaborn 提供的分布图绘制方法一般有这几个：
jointplot，pairplot，distplot，kdeplot。接下来，我们依次来看一下这些绘图方法的使用。
Seaborn 快速查看单变量分布的方法是 distplot。默认情况下，该方法将会绘制直方图并拟合核密度估计图。
"""

sns.distplot(iris["sepal_length"])



# distplot 提供了参数来调整直方图和核密度估计图，例如设置 kde=False 则可以只绘制直方图，或者 hist=False 只绘制核密度估计图。当然，kdeplot 可以专门用于绘制核密度估计图，其效果和 distplot(hist=False) 一致，但 kdeplot 拥有更多的自定义设置。
sns.kdeplot(iris["sepal_length"])

# jointplot 主要是用于绘制二元变量分布图。例如，我们探寻 sepal_length 和 sepal_width 二元特征变量之间的关系。
sns.jointplot(x="sepal_length", y="sepal_width", data=iris)


# jointplot 并不是一个 Figure-level 接口，但其支持 kind= 参数指定绘制出不同样式的分布图。例如，绘制出核密度估计对比图。
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="kde")




#六边形计数图：
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="hex")


#回归拟合图：
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="reg")


#最后要介绍的 pairplot 更加强大，其支持一次性将数据集中的特征变量两两对比绘图。默认情况下，对角线上是单变量分布图，而其他则是二元变量分布图。
sns.pairplot(iris)


#此时，我们引入第三维度 hue="species" 会更加直观。
sns.pairplot(iris, hue="species")




#回归图
#接下来，我们继续介绍回归图，回归图的绘制函数主要有：lmplot 和 regplot。
# regplot 绘制回归图时，只需要指定自变量和因变量即可，regplot 会自动完成线性回归拟合。
sns.regplot(x="sepal_length", y="sepal_width", data=iris)




#lmplot 同样是用于绘制回归图，但 lmplot 支持引入第三维度进行对比，例如我们设置 hue="species"。
sns.lmplot(x="sepal_length", y="sepal_width", hue="species", data=iris)



#矩阵图
#矩阵图中最常用的就只有 2 个，分别是：heatmap 和 clustermap。
#意如其名，heatmap 主要用于绘制热力图。
import numpy as np

sns.heatmap(np.random.rand(10, 10))



#热力图在某些场景下非常实用，例如绘制出变量相关性系数热力图。
#除此之外，clustermap 支持绘制层次聚类结构图。如下所示，我们先去掉原数据集中最后一个目标列，传入特征数据即可。当然，你需要对层次聚类有所了解，否则很难看明白图像多表述的含义。
iris.pop("species")
sns.clustermap(iris)










































print("=="*40)
print(" https://zhuanlan.zhihu.com/p/24464836 ")
print("=="*40)

#Histograms直方图
#直方图(Histogram)又称质量分布图。是一种统计报告图，由一系列高度不等的纵向条纹或线段表示数据分布的情况。 一般用横轴表示数据类型，纵轴表示分布情况。


import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt #导入

import seaborn as sns
sns.set(color_codes=True)#导入seaborn包设定颜色

np.random.seed(sum(map(ord, "distributions")))

x = np.random.normal(size=100)
sns.distplot(x, kde=False, rug=True);#kde=False关闭核密度分布,rug表示在x轴上每个观测上生成的小细条（边际毛毯）




#当绘制直方图时，你最需要确定的参数是矩形条的数目以及如何放置它们。利用bins可以方便设置矩形条的数量。如下所示：

sns.distplot(x, bins=20, kde=False, rug=True);#设置了20个矩形条

sns.distplot(x, hist=False, rug=True);#关闭直方图，开启rug细条


sns.kdeplot(x, shade=True);#shade控制阴影



#可以利用distplot() 把数据拟合成参数分布的图形并且观察它们之间的差距,再运用fit来进行参数控制。
x = np.random.gamma(6, size=200)#生成gamma分布的数据
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
d = rs.normal(size=100)

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



print("=="*40)
print(" https://wizardforcel.gitbooks.io/ds-ipynb/content/docs/8.17.html ")
print("=="*40)


import matplotlib.pyplot as plt
plt.style.use('classic')

import numpy as np
import pandas as pd


# 创建一些数据
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)




#并执行简单的绘图：
# 使用 Matplotlib 默认值绘制数据
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');





import seaborn as sns
sns.set()


#现在让我们重新运行与以前相同的两行：
# 和上面一样的绘图代码
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


#探索 Seaborn 绘图
#直方图，KDE，和密度
#通常在统计数据可视化中，你只需要绘制直方图和变量的联合分布。我们已经看到这在 Matplotlib 中相对简单：

data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col], alpha=0.5)


#我们可以使用核密度估计来获得对分布的平滑估计，而不是直方图，Seaborn 使用sns.kdeplot来执行：
for col in 'xy':
    sns.kdeplot(data[col], shade=True)

#直方图和 KDE 可以使用distplot组合：
sns.distplot(data['x'])
sns.distplot(data['y']);

#如果我们将完整的二维数据集传递给kdeplot，我们将获得数据的二维可视化：
sns.kdeplot(data['x'], data['y']);


#我们可以使用sns.jointplot查看联合分布和边缘分布。对于此图，我们将样式设置为白色背景：

with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde');


#还有其他参数可以传递给jointplot - 例如，我们可以使用基于六边形的直方图：
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')




#配对绘图
#将联合绘图推广到高维数据集时，最终会得到配对绘图。 当你想要绘制所有值对于彼此的配对时，这对于探索多维数据之间的相关性非常有用。
#我们将使用着名的鸢尾花数据集进行演示，该数据集列出了三种鸢尾花物种的花瓣和萼片的测量值：

iris = sns.load_dataset("iris")
print(f"iris.head() = {iris.head()}")


#可视化样本之间的多维关系就像调用sns.pairplot一样简单：
sns.pairplot(iris, hue='species', size=2.5);


#分面直方图
#有时，查看数据的最佳方式是通过子集的直方图。 Seaborn 的FacetGrid使其非常简单。我们将根据各种指标数据查看一些数据，它们显示餐厅员工在小费中收到的金额：
tips = sns.load_dataset('tips')
print(f"tips.head() = {tips.head()}")

tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15));


#因子图
#因子图也可用于此类可视化。 这允许你查看由任何其他参数定义的桶中的参数分布：

with sns.axes_style(style='ticks'):
    g = sns.factorplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill");

#联合分布
#与我们之前看到的配对图类似，我们可以使用sns.jointplot来显示不同数据集之间的联合分布以及相关的边缘分布：

with sns.axes_style('white'):
    sns.jointplot("total_bill", "tip", data=tips, kind='hex')


#联合图甚至可以做一些自动的核密度估计和回归：

sns.jointplot("total_bill", "tip", data=tips, kind='reg');



#条形图
#时间序列可以使用sns.factorplot绘制。 在下面的示例中，我们将使用我们在“聚合和分组”中首次看到的行星数据：

planets = sns.load_dataset('planets')
print(f"planets.head() = {planets.head()}")


with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=2,
                       kind="count", color='steelblue')
    g.set_xticklabels(step=5)



#通过查看每个行星的发现方法，我们可以了解更多信息：

with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',
                       hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')




#示例：探索马拉松结束时间
#在这里，我们将使用 Seaborn 来帮我们可视化和理解马拉松的结果。我从 Web 上的数据源抓取数据，汇总并删除任何身份信息，并将其放在 GitHub 上，可以在那里下载（如果你有兴趣使用 Python 抓取网页，我建议阅读 Ryan Mitchell 的《Web Scraping with Python》。我们首先从 Web 下载数据并将其加载到 Pandas 中：
# !curl -O https://raw.githubusercontent.com/jakevdp/marathon-data/master/marathon-data.csv

data = pd.read_csv('marathon-data.csv')
print(f"data.head() = {data.head()}")

#默认情况下，Pandas 将时间列加载为 Python 字符串（类型object）；我们可以通过查看DataFrame的dtypes属性来看到它：
print(f"data.dtypes = {data.dtypes}")

#让我们通过为时间提供转换器来解决这个问题：

def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return pd.datetools.timedelta(hours=h, minutes=m, seconds=s)

data = pd.read_csv('marathon-data.csv',
                   converters={'split':convert_time, 'final':convert_time})
print(f"data.head() = {data.head()}")
print(f"data.dtypes = {data.dtypes}")

#这看起来好多了。 出于我们的 Seaborn 绘图工具的目的，让我们接下来添加以秒为单位的列：

data['split_sec'] = data['split'].astype(int) / 1E9
data['final_sec'] = data['final'].astype(int) / 1E9
print(f"data.head() = {data.head()}")




#为了了解数据的样子，我们可以在数据上绘制一个jointplot：

with sns.axes_style('white'):
    g = sns.jointplot("split_sec", "final_sec", data, kind='hex')
    g.ax_joint.plot(np.linspace(4000, 16000),
                    np.linspace(8000, 32000), ':k')


#让我们在数据中创建另一个列，即分割分数，它测量每个运动员将比赛负分割或正分割（positive-split）的程度：
data['split_frac'] = 1 - 2 * data['split_sec'] / data['final_sec']
data.head()


#如果此分割差异小于零，则这个人将比赛以这个比例负分割。让我们绘制这个分割分数的分布图：

sns.distplot(data['split_frac'], kde=False);
plt.axvline(0, color="k", linestyle="--");




#让我们看看这个分割分数和其他变量之间是否存在任何相关性。我们将使用pairgrid来绘制所有这些相关性：

g = sns.PairGrid(data, vars=['age', 'split_sec', 'final_sec', 'split_frac'],
                 hue='gender', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend();







#看起来分割分数与年龄没有特别的关联，但确实与最终时间相关：更快的运动员往往将马拉松时间等分。（我们在这里看到，当涉及到绘图样式时，Seaborn 不是 Matplotlib 弊病的灵丹妙药：特别是，x轴标签重叠。因为输出是一个简单的 Matplotlib 图，但是，“自定义刻度”中的方法可以用来调整这些东西。）
#这里男女之间的区别很有意思。 让我们看看这两组的分割分数的直方图：

sns.kdeplot(data.split_frac[data.gender=='M'], label='men', shade=True)
sns.kdeplot(data.split_frac[data.gender=='W'], label='women', shade=True)
plt.xlabel('split_frac');



#这里有趣的是，有更多的男人比女人更接近等分！这几乎看起来像男女之间的某种双峰分布。 让我们看看，我们是否可以通过将分布看做年龄的函数，来判断发生了什么。
#比较分布的好方法是使用提琴图：

sns.violinplot("gender", "split_frac", data=data,
               palette=["lightblue", "lightpink"]);



#这是比较男女之间分布的另一种方式。
#让我们看得深入一些，然后将这些提琴图作为年龄的函数进行比较。我们首先在数组中创建一个新列，指定每个人的年龄，以十年为单位：

data['age_dec'] = data.age.map(lambda age: 10 * (age // 10))
print(f"data.head() = {data.head()}")


men = (data.gender == 'M')
women = (data.gender == 'W')

with sns.axes_style(style=None):
    sns.violinplot("age_dec", "split_frac", hue="gender", data=data,
                   split=True, inner="quartile",
                   palette=["lightblue", "lightpink"]);


#同样令人惊讶的是，这位 80 岁的女性在分割时间方面表现优于每个人。 这可能是因为我们估计来自小数字的分布，因为在该范围内只有少数运动员：
print(f"(data.age > 80).sum() = {(data.age > 80).sum()}")

#回到带有负分割的男性：谁是这些运动员？ 这个分割分数是否与快速结束相关？ 我们可以很容易地绘制这个图。 我们将使用regplot，它将自动拟合数据的线性回归：

g = sns.lmplot('final_sec', 'split_frac', col='gender', data=data,
               markers=".", scatter_kws=dict(color='c'))
g.map(plt.axhline, y=0.1, color="k", ls=":");


print("=="*40)
print(" https://zhuanlan.zhihu.com/p/40303932 ")
print("=="*40)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns





#2、图像审美
#这里当然不是说审美的标准是什么，而是如何调节图像的样式。这一块自由度太高就不具体介绍，就简单介绍一个修改背景的功能。

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
sinplot()
# 对两种画图进行比较
fig = plt.figure()
sns.set()
sinplot()



#主要有五种预设seaborn主题：darkgrid，whitegrid，dark，white，和ticks，利用set_style()来修改，不过这个修改是全局性的，会影响后面所有的图像。

sns.set_style('dark')
sinplot()


sns.set_style('whitegrid')
sinplot()







"""
3、可视化函数
对数据的可视化操作，乍看起来很复杂，其实种类总结起来可以分为下面几种：
1. 单变量分布可视化(displot)
2. 双变量分布可视化(jointplot)
3. 数据集中成对双变量分布(pairplot)
4. 双变量-三变量散点图(relplot)
5. 双变量-三变量连线图(relplot)
6. 双变量-三变量简单拟合
7. 分类数据的特殊绘图


"""


#4、单变量分布
#单变量分布可视化是通过将单变量数据进行统计从而实现画出概率分布的功能，同时概率分布有直方图与概率分布曲线两种形式。利用displot()对单变量分布画出直方图(可以取消)，并自动进行概率分布的拟合(也可以使用参数取消)。

sns.set_style('darkgrid')
x = np.random.randn(200)
sns.distplot(x);

sns.distplot(x,hist = False);


#5、双变量分布
#双变量分布通俗来说就是分析两个变量的联合概率分布和每一个变量的分布。

import pandas as pd
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df);

# 同样可以使用曲线来拟合分布密度
sns.jointplot(x="x", y="y", data=df, kind="kde");





#6、数据集中成对双变量分析
#对于数据集有多个变量的情况，如果每一对都要画出相关关系可能会比较麻烦，利用Seaborn可以很简单的画出数据集中每个变量之间的关系。

iris = sns.load_dataset("iris")
sns.pairplot(iris)

sns.pairplot(iris, hue='species', size=2.5)
# 对角线化的是单变量的分布


#7、双变量-三变量散点图
#统计分析是了解数据集中的变量如何相互关联以及这些关系如何依赖于其他变量的过程，有时候在对数据集完全不了解的情况下，可以利用散点图和连线图对其进行可视化分析，这里主要用到的函数是relplot函数。

tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips);

# 除了画出双变量的散点图外，还可以利用颜色来增加一个维度将点分离开
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);





# 为了强调数据之间的差异性，除了颜色也可以使用图形的不同来分类数据点（颜色和形状互相独立）
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",data=tips);



#8、双变量-三变量连续图
#为了进行数据分析，除了散点图，同样可以使用连续的线形来描述变化趋势。

df = pd.DataFrame(dict(time=np.arange(500),value=np.random.randn(500).cumsum()))
sns.relplot(x="time", y="value", kind="line", data=df);

# 可以选择不对x进行排序，仅仅需要修改sort参数即可
df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])
sns.relplot(x="x", y="y", sort=False, kind="line", data=df);



# 为了使线形更加的平滑可以使用聚合功能，表示对x变量的相同值进行多次测量，取平均，并取可信区间
fmri = sns.load_dataset("fmri")
plt.figure();
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri);
plt.figure();
sns.relplot(x="timepoint", y="signal",estimator=None,kind="line", data=fmri);

# 同时也可以使用颜色来区别不同种类
sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri);
# 也可以使用样式来区分，这里不加赘述


#9、简单线性拟合
#seaborn的目标是通过可视化快速简便地探索数据集，因为这样做比通过统计探索数据集更重要。用统计模型来估计两组噪声观察之间的简单关系可能会非常有用，因此就需要用简单的线性来可视化。

#线性模型可视化
#主要用regplot()进行画图，这个函数绘制两个变量的散点图，x和y，然后拟合回归模型并绘制得到的回归直线和该回归一个95％置信区间。

sns.set_style('darkgrid')
sns.regplot(x="total_bill", y="tip", data=tips);



sns.regplot(x="size", y="tip", data=tips);
# 对分类模型适应不好






#拟合不同类型的模型
#线性模型对某些数据可能适应不够好，可以使用高阶模型拟合

anscombe = sns.load_dataset("anscombe")
sns.regplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),ci=None);



# 利用order2阶模型来拟合
sns.regplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),ci=None,order = 2);



# 如果数据中有明显错误的数据点可以进行删除
sns.regplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),ci=None);
plt.figure()
sns.regplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),ci=None,robust = True);



#10、类型数据特殊绘图
#我们之前学习了如何使用散点图和回归模型拟合来可视化两个变量之间的关系。但是，如果感兴趣的主要变量之一类别的，那该怎么办？在这种情况下，散点图和回归模型方法将不起作用，就需要利用专门的分类可视化函数进行拟合。

#注：画图函数分为两种形式： 底层的类似于scatterplot(),swarmplot()来分别实现功能
#高层的类似于catplot()通过修改参数来实现上面底层的功能

#分类散点图
#可以使用两种方法来画出不同数据的分类情况，第一种是每个类别分布在对应的横轴坐标上，而第二种是为了展示出数据密度的分布从而将数据产生少量随即抖动进行可视化的方法。

# 微小抖动来展示出数据分布
sns.catplot(x="day", y="total_bill", data=tips);


# 利用jitter来控制抖动大小或者是否抖动
sns.catplot(x="day", y="total_bill", jitter = False,data=tips);




# 同时可以使用swarm方法来使得图形分布均匀
sns.catplot(x="day", y="total_bill", kind="swarm", data=tips);


# 值得注意的是，与上面的scatter相同，catplot函数可以使用hue来添加一维，但是暂不支持style
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);



#分类分布图
#随着数据的增加，分类数据的离散图更为复杂，这时候需要对每类数据进行分布统计。这里同样使用高级函数catplot()。

# 箱线图
# 显示了分布的三个四分位数值以及极值
sns.catplot(x="day", y="total_bill", kind="box", data=tips);

# 同样可以使用hue来增加维度
sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);



# 小提琴图事实上是密度图和箱型图的结合
# 分别表示箱型图的含义和任意位置的概练密度
sns.catplot(x="day", y="total_bill", hue="time",kind="violin", data=tips);






# 当hue参数只有两个级别时，也可以“拆分”小提琴，这样可以更有效地利用空间
sns.catplot(x="day", y="total_bill", hue="sex",kind="violin", split=True, data=tips);






#分类估计图
#如果我们更加关心类别之间的变化趋势，而不是每个类别内的分布情况，同样可以使用catplot来进行可视化。

# 条形图，利用bar来画出每个类别的平均值
# 黑色表示估计区间
titanic = sns.load_dataset("titanic")
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic);



# 如果更加关心的是类别的数量而不是统计数据的话可以使用count
sns.catplot(x="deck", kind="count", data=titanic);


# 点图功能提供了一种可视化相同信息的替代方式
# 只画出估计值和区间，而不画出完整的条形图
sns.catplot(x="sex", y="survived", hue="class", kind="point", data=titanic);




# 更加复杂的设置，通过对每一个参数进行设置来调节图像
sns.catplot(x="class", y="survived", hue="sex",
            palette={"male": "g", "female": "m"},
            markers=["^", "o"], linestyles=["-", "--"],
            kind="point", data=titanic);



print("=="*40)
print(" https://toutiao.io/posts/2a40rv/preview ")
print("=="*40)

"""
在本教程中我们主要讨论三种seaborn函数：

relplot(kind)  画图函数kind参数可以为"scatter"或者"line"

scatterplot()  散点图函数

lineplot()   折线图函数

不论数据结构多么复杂，seaborn的这些函数可以很简明的将数据展示出来。比如在二维空间中，seaborn可以在图中通过色调（hue）、尺寸（size）、风格（style）等方式来展示三维数据结构。具体怎么使用请看继续往下学习
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#让jupyter notebook的Cell可以将多个变量显示出来。
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

sns.set(style="darkgrid")



#使用散点图展现相关变量
#散点图是统计可视化方法中的中流砥柱。可以使用relplot(kind='scatter')或者scatter()作图

tips = sns.load_dataset("tips")
#tips数据集中包含total_bill、tip、sex、smoker、day、time、size这些字段
tips.head()
sns.relplot(x="total_bill", 
            y="tip", 
            data=tips)
#sns.scatterplot(x="total_bill", y="tip", data=tips) #效果等同于relplot函数


#style样式
#当有第三个变量参与绘图，在seaborn中可以使用style（类别样式参数），因为类别本身也是带有信息的。

sns.relplot(x="total_bill", 
            y="tip", 
            style="smoker", 
            data=tips)



#hue色标
#但是上图看着smoker类目，No和yes很难在图中区分。如果能有色调的区别，更容易肉眼区分开来。这里使用色调参数 hue

sns.relplot(x="total_bill", 
            y="tip", 
            hue="smoker", 
            style="smoker", 
            data=tips)





#当然我们也可以在图中对四种变量进行统计性分析。四个变量分别是total_bill、tip、smoker、time

sns.relplot(x="total_bill", 
            y="tip", 
            hue="smoker", 
            style="time", 
            data=tips)



#palette调色板
#在两个例子中，我们可以自定义调色板（color palette），这样做有很多options。在这里，我们使用字字符串接口自定义 顺序调色板。

sns.relplot(x="total_bill", 
            y="tip", 
            hue="size", 
            #ch变化范围(2.5,-0.2),从浅色（大于0）到深色（小于0）
            #色调的明暗程度dark取值范围（0，1），dark值越大散点的颜色越浅
            palette="ch:2.5,-0.2,dark=0.1", 
            data=tips)






#size尺寸
#对于三个变量的可视化，其实还可以使用尺寸size参数来可视化。

sns.relplot(x='total_bill', 
            y='tip', 
            size='size', 
            data=tips)


#size还可以设置范围

sns.relplot(x='total_bill',  
            y='tip', 
            size='size', 
            sizes=(15, 200),
            data=tips)




#散点图使用很方便，但这并不是万能通用的可视化方法。例如，我们想探索某些数据集中的某个变量随着时间的变化趋势，或者该变量是连续型变量，此时使用lineplot()更好。或者relplot(kind='line')。
#我们伪造一份随着时间变化的数据

data = dict(time=np.arange(500),
            value=np.random.randn(500).cumsum())

df = pd.DataFrame(data)

g = sns.relplot(x="time",
                y="value",
                kind="line",
                data=df)

g.fig.autofmt_xdate()




#sort参数
#因为lineplot()假定我们绘制的图因变量是y，自变量是x。绘图前默认从x轴方向对数据进行排序。但是，这种情况是可以被禁用的：

df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), 
                  columns=["x", "y"])

sns.relplot(x="x", 
            y="y", 
            sort=True, 
            kind="line", 
            data=df)

sns.relplot(x="x", 
            y="y", 
            sort=False, 
            kind="line", 
            data=df)









#03-聚合并表示不确定性
#更复杂的数据集将对x变量的相同值有多个观测值。seaborn中默认通过绘制x的平均值和x的95％置信区间来聚合每个x的多个测量值：

fmri = sns.load_dataset("fmri")
fmri.head()

sns.relplot(x='timepoint', 
            y='signal',
            kind='line',
            data=fmri)


#置信区间ci
#使用bootstrapping来计算置信区间（confidence intervals），这对于较大的数据集来说会是时间密集型的。因此我们可以对该方法采用禁用。参数ci=None表示禁用

sns.relplot(x='timepoint', 
            y='signal',
            kind='line',
            ci=None,
            data=fmri)



#另一个好的选择，特别是对于更大的数据，是通过绘制标准偏差sd而不是置信区间来表示每个时间点的分布范围。ci参数设置为'sd'

sns.relplot(x='timepoint', 
            y='signal',
            kind='line',
            ci='sd',
            data=fmri)




#estimator
#要完全剔除聚合效应，请将estimator参数设置为None。当数据在每个点上有多个观察值时，这可能会产生奇怪的效果。

sns.relplot(x='timepoint', 
            y='signal',
            kind='line',
            estimator=None,
            data=fmri)







"""
使用语义映射绘制数据子集
Plotting subsets of data with semantic mappings

lineplot()拥有与relplot()、scatterplot()类似的灵活性：同样可以借助色调hue、尺寸size和样式style将三种变量展示在二维图表中。因此我们不用停下来思考如何使用matplotlib对点线具体的参数进行设置。

使用lineplot()也会诊断数据如何借助语义进行聚合。例如在制图时，加入色调hue会将图表分为两条曲线以及错误带（error band），每种颜色对应的指示出数据的子集：

色调hue
下面我们看看hue具体例子


"""
fmri.sample(5)

sns.relplot(x='timepoint',
            y = 'signal',
            hue='event',
            kind='line',
            data=fmri)




#样式style
#改变制图中的样式

sns.relplot(x='timepoint',
            y = 'signal',
            hue='event',
            style='event',
            kind='line',
            data=fmri)








#注意上面代码中hue和style参数都是一个变量，所以绘制的图与之前生成的图变动不大。只是cue类曲线从实线变成虚线。

#现在hue和style参数不同后，我们在运行试试

sns.relplot(x='timepoint',
            y = 'signal',
            hue='region',
            style='event',
            kind='line',
            data=fmri)





#与散点图一样，要谨慎使用多个语义制作线图。 虽然这样操作有时候提供了信息，但图表更难解读。 但即使您只检查一个附加变量的变化，更改线条的颜色和样式也很有用。 当打印成黑白或有色盲的人观看时，这可以使情节更容易理解：
sns.relplot(x='timepoint',
            y = 'signal',
            hue='event',
            style='event',
            kind='line',
            data=fmri)






#units
#当您使用重复测量数据（即，您有多次采样的单位）时，您还可以单独绘制每个采样单位，而无需通过语义区分它们。这可以避免使图例混乱：

sns.relplot(x="timepoint", 
            y="signal", 
            hue="region",
            units="subject", 
            estimator=None,
            kind="line", 
            data=fmri.query("event == 'stim'"))







#lineplot（）中默认的色彩映射和图例处理还取决于色调hue是分类型数据还是数字型数据：
dots = pd.read_csv('dots.csv').query("align == 'dots'")
dots.head()

sns.relplot(x="time", 
            y="firing_rate",
            hue="coherence", 
            style="choice",
            kind="line", 
            data=dots)






#调色板palette
#可能会发生这样的情况：即使色调变量palette是数字，它也很难用线性色标表示。 这就是这种情况，其中色调变量hue以对数方式缩放。 您可以通过传递列表或字典为每一行提供特定的颜色值：

#n_colors值与coherence种类数相等
palette = sns.cubehelix_palette(light=.8, n_colors=6) 

sns.relplot(x="time", 
            y="firing_rate",
            hue="coherence", 
            style="choice",
            palette=palette,
            kind="line", data=dots);



#hue_norm
#或者您可以更改色彩映射的规范化方式

from matplotlib.colors import LogNorm

sns.relplot(x="time", 
            y="firing_rate",
            hue="coherence", 
            hue_norm=LogNorm(),
            style="choice",
            kind="line", 
            data=dots);



#size
#改变线条的粗细

sns.relplot(x='time',
            y='firing_rate',
            size='coherence',
            style='choice',
            kind='line',
            data=dots)





#绘制date数据
df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),
                       value=np.random.randn(500).cumsum()))
df.head()

g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()




#多图展现更多信息
#当你想要理解两个变量之间的关系如何依赖于多个其他变量时呢？

#最好的方法可能是制作一个以上的图。 因为relplot（）基于FacetGrid，所以这很容易做到。 要显示新增变量的影响，而不是将其分配给绘图中的一个语义角色，请使用它来“构思”（facet）可视化。 这意味着您可以创建多个轴并在每个轴上绘制数据的子集：

tips = pd.read_csv('tips.csv')

sns.relplot(x="total_bill", 
            y="tip", 
            hue="smoker",
            col="time",  #time有几种值，就有几列图
            data=tips);





sns.relplot(x="timepoint", 
            y="signal", 
            hue="subject",
            col="region",  #region有几种值，就有几列图
            row="event",  #event有几种值，就有几行图
            height=3,
            kind="line", 
            estimator=None, 
            data=fmri);



sns.relplot(x="timepoint", 
            y="signal", 
            hue="event", 
            style="event",
            col="subject", 
            col_wrap=3, #显示的图片的行数
            height=3, 
            aspect=1, #长宽比，该值越大图片越方。
            linewidth=2.5,
            kind="line", 
            data=fmri.query("region == 'frontal'"));



























print("=="*40)
print(" https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/716978/ ")
print("=="*40)
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




# https://zhuanlan.zhihu.com/p/27435863
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(sum(map(ord,"aesthetics")))  # 定义种子





def sinplot(flip=1):
    x = np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i*.5)*(7-i)*flip)




sinplot()

import seaborn as sns
sinplot()


"""
样式控制：axes_style() and set_style()
有5个seaborn的主题，适用于不同的应用和人群偏好：

darkgrid 黑色网格（默认）
whitegrid 白色网格
dark 黑色背景
white 白色背景
ticks 应该是四周都有刻度线的白背景？
网格能够帮助我们查找图表中的定量信息，而灰色网格主题中的白线能避免影响数据的表现，白色网格主题则类似的，当然更适合表达“重数据元素”（heavy data elements不理解）
"""


sns.set_style("whitegrid")
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data);



#对于许多场景，(特别是对于像对话这样的设置，您主要想使用图形来提供数据模式的印象)，网格就不那么必要了

sns.set_style("dark")
sinplot()




sns.set_style("white")
sinplot()




#有时你可能想要给情节增加一点额外的结构，这就是ticks参数的用途:

sns.set_style("ticks")
sinplot()
# 官方的例子在上方/右方也拥有刻度线，而验证时却没有（是jupyter notebook的原因？）




#用despine()进行边框控制
#white和ticks参数的样式，都可以删除上方和右方坐标轴上不需要的边框，这在matplotlib中是无法通过参数实现的，却可以在seaborn中通过despine()函数轻松移除他们。

sns.set_style("white")
sinplot() # 默认无参数状态，就是删除上方和右方的边框
sns.despine()


#一些图的边框可以通过数据移位，当然调用despine()也能做同样的事。当边框没有覆盖整个数据轴的范围的时候，trim参数会限制留存的边框范围。
f, ax = plt.subplots()
sns.violinplot(data=data)
sns.despine(offset=10, trim=True); # offset 两坐标轴离开距离；





#你也可以通过往despine()中添加参数去控制边框

sns.set_style("whitegrid")
sns.boxplot(data=data, palette="deep")
sns.despine(left=True) # 删除左边边框
st = sns.axes_style("darkgrid")



#临时设定图形样式
#虽然来回切换非常容易，但sns也允许用with语句中套用axes_style()达到临时设置参数的效果（仅对with块内的绘图函数起作用）。这也允许创建不同风格的坐标轴。

with sns.axes_style("darkgrid"):
    plt.subplot(211)
    sinplot()
plt.subplot(212)
sinplot(-1)

"""

seaborn样式中最重要的元素
如果您想要定制seanborn的样式，可以将参数字典传递给axes_style()和set_style()的rc参数。注意，只能通过该方法覆盖样式定义的一部分参数。(然而，更高层次的set()函数接受任何matplotlib参数的字典)。

如果您想要查看包含哪些参数，您可以只调用该函数而不带参数，这将返回当前设置的字典:

sns.axes_style()

{'axes.axisbelow': True,
 'axes.edgecolor': 'white',
 'axes.facecolor': '#EAEAF2',
 'axes.grid': True,
 'axes.labelcolor': '.15',
 'axes.linewidth': 0.0,
 'figure.facecolor': 'white',
 'font.family': ['sans-serif'],
 'font.sans-serif': ['Arial',
  'Liberation Sans',
  'Bitstream Vera Sans',
  'sans-serif'],
 'grid.color': 'white',
 'grid.linestyle': '-',
 'image.cmap': 'Greys',
 'legend.frameon': False,
 'legend.numpoints': 1,
 'legend.scatterpoints': 1,
 'lines.solid_capstyle': 'round',
 'text.color': '.15',
 'xtick.color': '.15',
 'xtick.direction': 'out',
 'xtick.major.size': 0.0,
 'xtick.minor.size': 0.0,
 'ytick.color': '.15',
 'ytick.direction': 'out',
 'ytick.major.size': 0.0,
 'ytick.minor.size': 0.0}
或许，你可以试试不同种类的参数效果
"""

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sinplot()





#通过 plotting_context() 和 set_context() 调整绘图元素
#另一组参数控制绘图元素的规模，这应该让您使用相同的代码来制作适合在较大或较小的情节适当的场景中使用的情节。

#首先，可以通过sns.set()重置参数。

sns.set()
#四种预设，按相对尺寸的顺序(线条越来越粗)，分别是paper，notebook, talk, and poster。notebook的样式是默认的，上面的绘图都是使用默认的notebook预设。

sns.set_context("paper")
plt.figure(figsize=(8,6))
sinplot()


# default 默认设置
sns.set_context("notebook")
plt.figure(figsize=(8,6))
sinplot()

sns.set_context("talk")
plt.figure(figsize=(8,6))
sinplot()


sns.set_context("poster")
plt.figure(figsize=(8,6))
sinplot()




#通过观察各种样式的结果，你应当可以了解context函数

#类似的，还可以使用其中一个名称来调用set_context()来设置参数，您可以通过提供参数值的字典来覆盖参数。

#通过更改context还可以独立地扩展字体元素的大小。(这个选项也可以通过顶级set()函数获得）。

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sinplot()
































































































