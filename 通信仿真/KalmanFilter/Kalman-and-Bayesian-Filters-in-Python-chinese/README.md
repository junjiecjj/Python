# [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)


这是一本卡尔曼滤波器和贝叶斯滤波器入门教材。这本书使用Jupyter Notebook编写，因此你可以在浏览器中阅读这本书，还可以运行和修改书中的代码并实时查看结果。还有比这更好的学习方法吗？


**"Kalman and Bayesian Filters in Python" 看起来非常惊人! ... 你这本书正是我所需要的** - Allen Downey, Professor and O'Reilly author.

**感谢您在发布卡尔曼滤波器的入门教程以及Python卡尔曼滤波器库方面所做的所有工作。我们一直在内部用它给大伙教授一些重要的状态估计概念，它真是帮了一个大忙。** - Sam Rodkey, SpaceX


点击下面的binder或Azure badge以开始在线阅读:


[![Binder](http://mybinder.org/badge.svg)](https://beta.mybinder.org/v2/gh/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master)
<a href="https://notebooks.azure.com/import/gh/rlabbe/Kalman-and-Bayesian-Filters-in-Python"><img src="https://notebooks.azure.com/launch.png" /></a>



![alt tag](https://raw.githubusercontent.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/animations/05_dog_track.gif)

什么是卡尔曼与贝叶斯滤波器?
-----

所有传感器都有噪声。这个世界充满了各种我们想要测量和跟踪的数据与事件，但我们并不能依靠传感器来得到完美准确的信息。我车上的GPS显示高度，每次我经过同一个位置，它报告的高度略有不同。如果我使用厨房秤称同一个物体的重量两次，会得到不同的读数。

在简单的情况下，解决方案是显而易见的，如果测量仪器给出的读数只是稍有不同，我可以多读几次然后取其平均值，或者我可以使用更高精度的测量仪器。但是，但传感器噪声很大，或者在某些环境下采集数据比较困难时，我们该怎么做呢？我们可能想尝试跟踪一架低空飞行器的飞行轨迹，也可能想给无人机设计一个自动驾驶仪，亦或者想确保农场的拖拉机能够毫无遗漏地播种整个田地。我的工作与计算机视觉有关，需要跟踪图像中的运动物体，而计算机视觉算法会产生充满噪声且不太可靠的结果。

这本书教你如何解决这些滤波问题。我使用了许多不同的算法，但它们均基于贝叶斯概率。简单地说，贝叶斯概率根据过去的信息来判断未来可能发生的事。

如果我让你说出我车子现在的方向，你可能会一脸懵逼。你最多只能提供一个介于1到360°之间的数字，那么只有1/360的机会是正确的。现在假设我告诉你，2秒前我车子的航向是243°，而2秒内车子的方向变化不太可能会很大，因此你便能得出一个更精确的预测。你用过去的信息更准确地推断出当前或未来的信息。

这个世界也是充满噪声的。上面的预测有助于你做出更好的估计，但它也会受到噪声的影响。我可能会为了一条狗而突然刹车，或者绕着一个坑洞转弯。路上的强风和结冰是影响我汽车行驶路径的外部因素。在控制相关文献中，我们称这些为噪声，尽管你可能不会这么认为。

关于贝叶斯概率还不止这些，但可能你已经了解其主要概念了。知识是不确定的，我们根据现象的强度来改变我们的置信度。卡尔曼与贝叶斯滤波器将我们对系统如何运行的嘈杂且有限的认知和同样嘈杂且有限的传感器测量结合起来，以得到对系统状态的最优估计。我们的原则是永远不要丢弃信息。

假设我们正在追踪一个物体，传感器告诉我们它突然改变方向了。是真的改变了，还只是传感器数据有噪声？这得看情况而定，如果这时一架喷气式战斗机，我们会非常倾向于相信它的确发生了一个瞬时机动，而如果这是一辆直线轨道上的货运列车，那我们会降低数据的可信度，同时我们会进一步根据传感器的精度来修正它的置信度。也就是说，置信度取决于我们对正在跟踪的系统的认知程度以及传感器特性。

卡尔曼滤波器最初是由Rudolf Emil Kálmán发明的，用数学上的最优方法来解决这类问题。它最初用于阿波罗登月任务，从那时起，它就被广泛用于各种领域。飞机、潜艇和巡航导弹上都使用了卡尔曼滤波器。它们被华尔街用于跟踪金融市场，还被用于机器人、物联网传感器和实验室仪器中。化工厂用它来控制和监测化学反应，医学成像中用于去除心脏信号中的噪声。在涉及传感器和时间序列数据的应用中，通常都会运用到卡尔曼滤波器或者相似方法。

动机
-----

写下这本书的动机来源于我曾经对一个友好的卡尔曼滤波教程的渴望。我是一名软件工程师，在航空航天领域工作了近20年，所以我经常会在卡尔曼滤波器上碰壁，但却从未实现过一个，毕竟它们是出了名的困难。这个理论很优美，但是如果你在信号处理、控制理论、概率和统计以及制导和控制理论等方面还没有比较好的训练的话，学习起来就很困难。随着我开始使用计算机视觉来解决跟踪问题，自己来实现卡尔曼滤波器的需求也变得迫切了。

这个领域并不缺少优秀的教科书，如Grewal和Andrew的《Kalman Filtering》。但是，如果不了解一些必要的背景知识，那坐下来直接阅读这些书籍只会让你感到沮丧和厌烦。一般来说，前几章会概述需要数年学习的大学数学课程内容，轻描淡写地让你参考有关微积分的教科书，并在几个简短的段落中展示需要整个学期来学习的结论。它们是高年级本科或研究生课程的教科书，也是研究人员和专业人士的宝贵参考资料，但对于更一般的读者来说，确实学习起来会很困难。引入的数学符号没有注释，不同的文档中可能会使用不同的单词和变量名来表示同一个概念，并且这些书中几乎没有实例或解决一些实际问题。我经常发现自己能看懂每一个单词并了解那些数学定义，却完全不能理解这些单词和数学公式试图描述的真实世界现象是什么，我会反复思考：“但这些是什么意思呢？”。

然而，当我终于理解卡尔曼滤波器后，才意识到这些基本概念是非常简单的。如果你熟悉一些简单的概率定理，并且对如何融合不确定事物有一定直觉，那便能理解卡尔曼滤波器的概念。卡尔曼滤波器以困难著称，但是抛弃很多专业术语后，我愈发清晰看到它本质和数学上的美丽，于是我爱上了它。

更多的困难出现在我尝试开始去理解这些数学和理论的时候，一本书或者一篇论文会去陈述事实并提供图表以证明，但不幸的是，我仍然不清楚这个陈述为什么是正确的，又或者我无法重现其图表。或有时我想知道“这是否成立如果R=0?”，作者可能会提供一些比较高层次的伪代码以至于无法轻易实现。有些书提供Matlab代码， 但可惜我没有使用这种昂贵商业软件所需要的许可证。另外 ，还有很多书在每一章末尾提供了许多有用的练习，如果你想要自己实现卡尔曼滤波器，那么便要去理解这些没有答案的练习。如果你在课堂上使用这本书，那可能关系不大，但对于独立读者来说这便很糟糕了，我讨厌作者对我隐瞒信息，大概他是为了避免学生在课堂上作弊吧。

所有的这些都会阻碍学习。我想在屏幕上追踪一张图像，或者为我的Arduino项目写一些代码，因此我想知道这些书中的图表是如何产生的，并且想选择一些与作者所提供的不一样的参数。我想仿真运行，看看滤波器在信号中加入更多噪声时是如何表现的。在每天的代码中都有成千上万使用卡尔曼滤波器的机会，而这些相当直观简单的项目也是火箭科学家和学者诞生的源头。

我写这本书便是为了满足所有上述需求。但如果你在设计军用雷达，这并不能成为你唯一的参考资料，你更需要的是去一所很棒的STEM学校攻读硕士或博士学位。这本书提供给那些需要滤波或平滑一些数据的爱好者，好奇者和正在工作的工程师。如果你是一个爱好者，这本书应该能提供你所需要的一切。如果你想深入专研卡尔曼滤波器，那你还需要学习更多，我意图介绍足够的概念和数学基础，让你更容易去学习教科书和论文。

这本书是交互式的，虽然你也可以把它当成静态内容直接在线阅读，但我强烈建议你按照预期来使用它。基于Jupyter Notebook，让我可以在书中任何地方组织文本、数学公式、Python代码和其输出。本书中每一个数据和图表都是由notebook中的Python生成的。想成倍修改某个参数值？只需要更改参数的值，然后点击“运行”，便会生成新的图表或者打印输出。

这本书有练习，但也都有答案。我信任你。如果你只是需要一个答案，那就直接阅读答案。而如果你想更深刻理解这些知识，那么在这之前先尝试去实现那些练习吧。因为这本书具有交互性，你可以直接在本书中输入并运行你的答案，而不必要迁移到其它不一样的环境，也不用在开始前导入大量内容。

这本书包含了统计数据计算、滤波器相关数据绘制及各种我们会涉及到的滤波器所需的支持库。这仍需要一个强烈的警告：大多数代码是为教学目的而编写的，我很少选择最高效的解决方案（这常常掩盖了代码的真实意图），而且在书的第一部分，我并不关心数值稳定性。理解这一点很重要——飞机上的卡尔曼滤波器经过精心设计并且其实现也是稳定的。在许多情况下，原始的实现并不稳定。如果你想很认真地学习卡尔曼过滤器，这本书将不会是最后一本你需要的书。我的目的是向你们介绍概念和数学，并让你们达到能学习教科书的程度。

最后，这本书是免费的。我曾经花了数千美元去购买卡尔曼相关书籍，这对于那些手头拮据的学生来说几乎是不可能的事情。 我曾从诸如Python之类的免费软件和[Allen B. Downey ](http://www.greenteapress.com/)的那些免费书籍中获益颇多，是时候回报了。因此，这本书是免费的，它托管在Github的免费服务器上，并且只用到了免费和开源的软件如IPython和MathJax。


## 在线阅读

这本书编写成了Jupyter Notebooks的一个集合，这是一个基于浏览器的交互式系统，允许你将文本、Python和数学结合到浏览器中。下面列出了多种在线阅读方式。

### binder

binder提供notebooks在线交互服务，所以你可以直接在浏览器上修改和运行代码，而不必下载这本书或者安装Jupyter。

[![Binder](http://mybinder.org/badge.svg)](https://beta.mybinder.org/v2/gh/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master)


### nbviewer

网站http://nbviewer.org 提供了一个Jupyter Notebook服务器，可以转换存储在Github（或其它地方）上的notebooks。你可以使用 [*this nbviewer link*](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb) 通过nbviewer来访问我这本书。如果你在今天阅读我这本书，然后我明天修改了某处，明天阅读时你将看到这个修改。Notebooks的转换是静态的 - 你只能阅读，而无法修改或运行代码。

nbviewer 似乎需要数天来更新内容版本，所以你读到的不一定是最新版的。

### GitHub

Github可以直接转换notebooks。这是最快捷的一种方式，你只需要点击上面的文件就可以了。不过它转换的数学公式有些不准确，如果你不仅仅是想单纯地读书，我不太建议使用它。

PDF 版本
-----

这里有一个可用的PDF版本 [here](https://drive.google.com/open?id=0By_SW19c1BfhSVFzNHc0SjduNzg)

PDF中的内容通常会落后于Github上的，因为我不是每次都会去更新它。

## 下载和运行这本书

不管怎样，这本书设计成了交互式的，我也建议以这种形式来使用它。做成这样花费了一番功夫，但这是很值得的。如果你在电脑上安装了IPython和一些支持库，那可以把这本书clone下来，你将能够自己运行书中的所有代码。你可以执行实验，观察滤波器对于不同数据的表现，等等。我发现这种即时反馈不仅很重要而且会让人充满动力，你不用去怀疑“如果会怎样”，放手去试吧！

可以在终端中运行以下命令以获取这本书及相关支持软件：

    git clone --depth=1 https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.git
    pip install filterpy

IPython环境的安装搭建介绍可以在安装附录中找到，见此处 [here](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-A-Installation.ipynb).

安装软件后，你可以进入到安装目录并在终端中通过命令行运行juptyer notebook

    jupyter notebook

它将会打开一个浏览器窗口，显示根目录的内容。这本书分成许多章，每个章节都命名为xx-name.ipynb，其中xx是章节号，.ipynb是Notebook文件的扩展名。要阅读第二章，请单击第二章的链接，浏览器将会打开那个子目录。每个子目录中将会有一个或多个IPython Notebooks（所有notebooks都有.ipynb文件扩展名）。章节内容在notebook中，与章节名称同名。每章中还会有许多实现相关功能如生成显示动画的supporting notebooks，每个用户没必要去阅读这些，但如果你对于动画是如何制作的比较好奇，那就看一眼吧。

诚然，对于一本书来说这个界面过于繁琐。我正在跟随其它几个项目的脚步，这些项目将Jupyter Notebook重新设计以生成完整的一本书。我觉得这些繁琐的事情会有巨大的回报——当你读一本书的时候，不必下载一个单独的代码库并在一个IDE中运行它，所有的代码和文档都会在同一个地方。如果你想修改代码，你可以立即这么做并即时看到修改的效果。你可以修复一个你发现的bug，然后推送到我的仓库中，这样全世界的人们都能受益。当然，你也永远不会遇到我在传统书籍中一直需要面临的问题——书籍和其代码彼此不同步，你需要绞尽脑汁去判断哪个来源更可靠些。


配套软件
-----

[![Latest Version](http://img.shields.io/pypi/v/filterpy.svg)](http://pypi.python.org/pypi/filterpy)

我写了一个名为**FilterPy**的开源贝叶斯滤波python库。我已经在PyPi（Python包目录）上提供了这个项目。要从PyPi安装，请在终端中输入命令

    pip install filterpy

如果你还没有pip, 可以按照一下说明操作: https://pip.pypa.io/en/latest/installing.html.

All of the filters used in this book as well as others not in this book are implemented in my Python library FilterPy, available [here](https://github.com/rlabbe/filterpy). You do not need to download or install this to read the book, but you will likely want to use this library to write your own filters. It includes Kalman filters, Fading Memory filters, H infinity filters, Extended and Unscented filters, least square filters, and many more.  It also includes helper routines that simplify the designing the matrices used by some of the filters, and other code such as Kalman based smoothers.


FilterPy is hosted github at (https://github.com/rlabbe/filterpy).  If you want the bleading edge release you will want to grab a copy from github, and follow your Python installation's instructions for adding it to the Python search path. This might expose you to some instability since you might not get a tested release, but as a benefit you will also get all of the test scripts used to test the library. You can examine these scripts to see many examples of writing and running filters while not in the Jupyter Notebook environment.

Alternative Way of Running the Book in Conda environment
----
If you have conda or miniconda installed, you can create environment by

    conda env update -f environment.yml

and use

    source activate kf_bf

and

    source deactivate kf_bf

to activate and deactivate the environment.


Issues or Questions
------

If you have comments, you can write an issue at GitHub so that everyone can read it along with my response. Please don't view it as a way to report bugs only. Alternatively I've created a gitter room for more informal discussion. [![Join the chat at https://gitter.im/rlabbe/Kalman-and-Bayesian-Filters-in-Python](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/rlabbe/Kalman-and-Bayesian-Filters-in-Python?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


License
-----
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Kalman and Bayesian Filters in Python</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python" property="cc:attributionName" rel="cc:attributionURL">Roger R. Labbe</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

All software in this book, software that supports this book (such as in the the code directory) or used in the generation of the book (in the pdf directory) that is contained in this repository is licensed under the following MIT license:

The MIT License (MIT)

Copyright (c) 2015 Roger R. Labbe Jr

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.TION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contact
-----

rlabbejr at gmail.com
