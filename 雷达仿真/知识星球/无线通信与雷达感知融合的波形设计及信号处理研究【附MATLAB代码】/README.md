# BachelorThesis
ZJU Bachelor Thesis

这些程序基本上是试图复现这篇文章所用的通感方案：

Pucci L, Paolini E, Giorgetti A. System-Level Analysis of Joint Sensing and Communication Based on 5G New Radio[J]. IEEE Journal on Selected Areas in Communications, 2022, 40(7): 2043–2055.

不过上述文章的重点是根据这个方案对通感一体化做系统性的性能分析，方案本身是很简陋的，而且很早就在下面这篇文章中被提出了：

Sturm C, Wiesbeck W. Waveform Design and Signal Processing Aspects for Fusion of Wireless Communications and Radar Sensing[J]. Proceedings of the IEEE, 2011, 99(7): 1236–1259.

（虽然我不确定这就是最早的提出这个简陋方案的文章）



main.m是蒙特卡洛运行很多次，统计检测概率和误差

main＿noMC.m是不进行蒙特卡洛实验，只对论文里用到的一组效果不错的数据运行一次

main_DoAComparison.m是对改进前后的DoA效果进行对比
