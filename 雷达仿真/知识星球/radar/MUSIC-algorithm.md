---
layout: post
title: MUSIC算法
date: 2024-05-15 17:59:03
excerpt_type: html
categories: [信号处理算法]
---

MUSIC算法是一种基于子空间分解的频谱方法，通过对信号子空间进行分析，能够准确地估计出多个信号的到达方向。
<!-- more -->

## MUSIC(Multiple Signal Classification)算法概述

[Matlab文档-MUSIC Super-Resolution DOA Estimation](https://ww2.mathworks.cn/help/phased/ug/music-super-resolution-doa-estimation.html)

[Wiki-MUSIC (algorithm)](https://en.wikipedia.org/wiki/MUSIC_(algorithm))

MUSIC算法是一种基于在阵列上观察到的传感器协方差矩阵的特征值分解的高分辨率方向搜索算法。MUSIC 属于基于子空间的方向搜索算法家族。

MUSIC的基本原理可以概括为：根据传感器阵列可以得到到达矢量(arrival vector) $a(\omega)$，多个目标的到达矢量组成到达矢量矩阵 $A$，根据多个通道的接受信号得到样本传感器协方差矩阵,从样本传感器协方差矩阵中估计传感器协方差矩阵 $R_{x}$，计算特征向量得到信号子空间  $\mathcal{U_{s}}$ 和噪声子空间  $\mathcal{U_{n}}$ 的基。  $\omega$ ,因此任意一个**到达矢量 $a(\omega)$ 和噪声子空间的基正交**。遍历 $\omega$ ，当 $\omega$ 和待检测目标角度一致时 $|a^{H}(\omega)U_{N}|$  得到最小值。

## 信号模型

假设存在 $p$ 个目标和 $M$ 个传感器（包括MIMO得到的虚拟通道）

$$
\begin{equation}
    x(n)=As(n)+n(n)  
\end{equation} 
$$

$$
\begin{equation}A=[a(\omega)_1,a(\omega)_2,...a(\omega)_p]\end{equation} 
$$

$$
\begin{equation}s(n)=[s_1(n),s_2(n),...,s_p(n)]'\end{equation} 
$$

- $x(n)$： $M×1$ 矩阵传感器接收到的（离散）信号，是信号处理时的已知量
- $A$： $M×p$ 矩阵，到达矢量(arrival vector)矩阵，，由每个目标的导向矢量组合而成
- $s(n)$： $p×1$ 矩阵，表示信号源的信号，假设不同信号源的信号互不相关
- $n(n)$： $M×1$ 矩阵，噪声，假设噪声之间和噪声与信号之间都互不相关

### 到达矢量

到达矢量用于表示


$$
\mathbf{a}(\theta) = \begin{bmatrix}
1 \\
e^{-j \frac{2\pi d}{\lambda} \sin(\theta)} \\
e^{-j \frac{2\pi d}{\lambda} 2 \sin(\theta)} \\
\vdots \\
e^{-j \frac{2\pi d}{\lambda} (M-1) \sin(\theta)}
\end{bmatrix}
$$


## 信号子空间与噪声子空间

计算自相关矩阵

$$
R_{xx}=E\{ xx^{H}\}=\frac{1}{N}\sum_{n=1}^{N}x(n)x(n)^{H} 
$$

将式 (1) 代入得：

$$
\begin{align}
R_{xx}&=E\{(As(n)+n(n))(As(n)+n(n))^{H}\}  \qquad, s(n)\text{与} n(n) \text{不相关} \\  
&=AR_{ss}A^{H}+\sigma^{2}I  
\end{align}
$$

可见 ${R_{xx}}$ 为厄尔米特(Hermitian)矩阵，令其特征值分解为：

$$
R_{xx}=U\Sigma U^H  
$$

其中 $U$为特征向量矩阵，每一列为一个特征向量， $U=[v_1,v_2,...,v_M]$， $\Sigma$为特征值矩阵（对角线上为特征向量对应的特征值，其余位置为0）。

由于 $rank(AR_{ss}A^{H})=p < M$ ，对构建有效信号 $AR_{ss}A^{H}$ 做出主要贡献的特征向量实际上只有 $p$ 个，其余特征向量张成噪声子空间。 将  代入 得到：

$$
\begin{align}
R_{xx}&=AR_{ss}A^{H}+\sigma^{2}UU^{H}    \\ 
U^{H}R_{xx}U&=U^{H}AR_{ss}A^{H}U+\sigma^{2}I_{M×M}\\
\Sigma&=diag(\alpha_1^2,...,\alpha_p^2,0,...,0)+\sigma^{2}I_{M×M}\\
\Sigma&=diag(\alpha_1^2+\sigma^{2},...,\alpha_p^2+\sigma^{2},\sigma^{2},...,\sigma^{2})
\end{align}
$$

当信噪比较高时，可以从取出 ${p}$个特征值较大的特征向量组成 $U_s=[v_1,v_2,...,v_p]$，上下几个特征值较小的特征向量组成 $U_n=[v_{p+1},...,v_M]$。分别可以张成信号子空间   $\mathcal{U_{S}}$和噪声子空间 $\mathcal{U_{n}}$ 。用 $[U_s,U_n]$替换 $U$可以得到：

$$
\begin{align}
R_{xx}=[U_s,U_n]\Sigma [U_s,U_n]^H  
R_{xx}U_n=[U_s,U_n]\Sigma  
    \begin{bmatrix}
        0 \\
        I
    \end{bmatrix}\\
   R_{xx}U_n =\sigma^2U_n
\end{align}
$$

$$
\begin{align}
R_{xx}=AR_{ss}A^{H}+\sigma^{2}UU^{H}   \\
R_{xx}U_n=AR_{ss}A^{H}U_n+\sigma^2U_n   
\end{align}
$$

由(8)和(9)得： $AR_{ss}A^{H}U_n=0$，即

$$
A^{H}U_n=\mathbf{0} 
$$

也就是说任意一个目标对应的到达矢量 $a(\omega_p)$都与噪声特征向量矩阵 $U_n$ 正交。

$$
a^{H}(\omega)U_n=\mathbf{0}, a(\omega) \in A
$$




## MUSIC伪谱（MUSIC pseudospectrum）

根据式(11)，我们可以遍历可能的角度，从而搜索搜索所有与噪声子空间正交的到达向量。为了进行搜索，MUSIC构建了一个依赖于到达角度的功率表达式，称为MUSIC伪谱：


$$
P_{MUSIC}(\omega) = \frac{a^{H}(\omega) a(\omega)}{|| a(\omega)^{H} U_{n}  ||^{2} } =  \frac{a^{H}(\omega) a(\omega)}{a^{H}(\omega)  U_{n}  U_{n}^{H}  a(\omega)}
$$
