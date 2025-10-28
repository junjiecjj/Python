---
layout: page
title: 到达矢量 (Arrival vector)
date: 2024-05-19 9:55:00
categories: [信号处理算法]
---

在阵列信号处理中，常常使用到达矢量来描述同一个信号被不同天线接收到时的差距——时间差、相位差。波束形成和波达方向估计都需要用到达矢量。
<!-- more -->

::: tip
到达矢量(arrival vector)和导向矢量(steering Vector)的形式的计算形式是一样的，只是用途不一样。到达矢量一般是根据实际接收到的来自特定方向的信号测量或估计，导向矢量是根据希望的方向计算，以将阵列引导到特定方向。
::: 

# 接收信号模型


阵列天线系统的输入信号 $\boldsymbol{x}(n)$ 常常使用以下模型

$$
\begin{equation}
    \boldsymbol{x}(n)=\boldsymbol{A} \boldsymbol{s}(n)+ \boldsymbol{n}(n)  
\end{equation}
$$

- $\boldsymbol{x}(n)$：   表示系统的输入信号， $\boldsymbol{x}(n)$ 是一个 $M\times 1$ 的列向量，分别表示 $M$ 个阵元接收到的信号。

- $\boldsymbol{s}(n)$：   并表示信号源发送过来的信号。一般选择其中1个阵元作为参考阵元，$\boldsymbol{s}(n)  = \begin{bmatrix}  s_{1}(n) \\ s_{2}(n) \\ \vdots \\ s_{p}(n) \end{bmatrix}$  ,  $s_{i}(n)$ 表示参考阵元接收到的第 $i$ 信号源的信号。

- $\boldsymbol{A}$:    是一个 $M\times p$ 的到达矢量矩阵，每一列都是一个到达矢量(arrival vector)，$\boldsymbol{A}=[\boldsymbol{a}(\omega_{1}),\boldsymbol{a}(\omega_{2}), \cdots ,\boldsymbol{a}(\omega_{p})]$，用于表示不同阵元接收到的信号的差异，通常 $x_{1}(n)$ 是参考阵元的接受信号， 此时 $A$ 的第一行为 $[1,1,\cdots,1]$ 。


 

- $\boldsymbol{n}(n)$:    表示噪声，是$M\times 1$ 的列向量。


# 单个目标的到达矢量

首先建立坐标系，令$x$ 轴正方向为天线的朝向， $xoy$ 平面上与 $x$ 轴正方向的夹角为方位角 $\theta$ ， $y$ 轴正方向 $\theta = 90 \degree$ ；与 $z$ 轴正方向的夹角为俯仰角 $\phi$ 。
 
先考虑单信号源、两个阵元的情况。

假设**信号源**发射的是**窄带信号**$s_1(t)$，其中心频率为 $f_{c}$ ；

>之所以假设信号是窄带信号，是因为在信号带宽较窄的情况下，信号的频率成分集中，相邻频率分量的相位差距随时间变化相对缓慢，从而在空间阵列上的时间延迟可以近似的使用相位差来表示。

假设信号源相对于阵列的方向为 $\boldsymbol{\omega_{1}} = [\theta,\phi]$ ，且信号源与阵列的距离够远，可以视为平行信号源。
>天线的间距一般为 $\frac{\lambda}{2}$，也就是说24G毫米波雷达的天线间距为6.2mm，相比于待测物体的距离是一个很小的值。

将其中一个阵元设为**参考阵元**接收到的信号为 $x_{1}(t) = 1 \times s_1(t)$ 。 另一个阵元的接收信号为  $x_{2}(t)$ ，与参考阵元的相对位置为  $\boldsymbol{r}_2'= \begin{bmatrix}  x \\ y \\ z \end{bmatrix}$ 。


当前信号源方向的单位向量 $\boldsymbol{e}_{1} = \begin{bmatrix}  \cos{\phi} \sin{\theta} \\ \cos{\phi} \cos{\theta} \\ \sin{\phi} \end{bmatrix}$ 

信号到达该阵元相比参考阵元少走的距离为 $d = \langle \boldsymbol{r}_2', \boldsymbol{e}_{1} \rangle$

也就是说，信号到达该阵元相比参考阵**早** $\tau = \frac{d}{c}$ 同一时刻，该阵元接收到的信号是参考阵元 $\tau$ 秒后的信号，即 $x_{2}(t) = x_{1}(t+\tau)$

上文已经提到过，对于窄带信号来说，可以用相位差近似表示时间差。举一个极端的例子

$$
\begin{align}
s(t) &= e^{-j2\pi ft} \\
s(t-\tau ) &= e^{-j2\pi f(t-\tau)} = e^{-j2\pi ft} \cdot e^{j2\pi f \tau} = s(t) \cdot e^{j2\pi f \tau}
\end{align}
$$

近似的,我们把载波频率直接当作窄带信号的频率，可以得到该阵元和参考阵元接收信号的关系：

$$
\begin{align}
    x_{2}(t) &= x_{1}(t) \cdot e^{j2\pi f \tau} \\
    x_{2}(t) &= x_{1}(t) \cdot \exp({j2\pi \frac{\langle \boldsymbol{r}_2', \boldsymbol{e}_{1} \rangle}{\lambda}})
\end{align}
$$

因此，阵列的接收信号可以近似的表示为：

$$ 
\begin{equation}
\boldsymbol{x}(t) = 
\begin{bmatrix}
    x_{1}(t)\\ 
    x_{2}(t)
\end{bmatrix}
=  
\begin{bmatrix}
    1\\ 
    \exp({j2\pi \frac{\langle \boldsymbol{r}_2', \boldsymbol{e}_{1} \rangle}{\lambda}})
\end{bmatrix}
s_1(t)
\end{equation}
$$

# 多目标多阵元情况下

令 $\boldsymbol{a}(\omega_{n})=\begin{bmatrix}  1\\ \exp({j2\pi \frac{\langle \boldsymbol{r}_2', \boldsymbol{e}_{n} \rangle} {\lambda}}) \\ \vdots \\ \exp({j2\pi \frac{\langle \boldsymbol{r}_M', \boldsymbol{e}_{n} \rangle} {\lambda}}) \end{bmatrix}_{M \times 1}$ , 表示第n个信号源在包含M个阵元的阵列上的到达矢量。 $\boldsymbol{a}(\omega_{n})s_n(t)$ 可以表示该信号源在阵列上的接收信号，不同信号源之间使用加法叠加。

假设有空间中共有 $p$ 个信号源，其到达矢量沟通组成一个 $M \times p$ 的矩阵 $A$ ，称为**到达矢量矩阵**，其中 $A = \begin{bmatrix} \boldsymbol{a}(\omega_{1}) , \boldsymbol{a}(\omega_{2} ) , \dots , \boldsymbol{a}(\omega_{p}) \end{bmatrix}$ 

添加加性噪声后，得到阵列接收信号

$$
\begin{equation}
    \boldsymbol{x}(t)=\boldsymbol{A} \boldsymbol{s}(t)+ \boldsymbol{n}(t)  
\end{equation}
$$

# 均匀线阵的到达矢量

假设现在又一个包含M个阵元的均匀线阵， 相邻两个阵元间隔 $\frac{\lambda}{2}$ ， 其坐标分别为 $\begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$ ,$\begin{bmatrix} 0  \\ 0\\ \frac{\lambda}{2} \end{bmatrix}$ ,$\begin{bmatrix} 0 \\ 0\\ \frac{2\lambda}{2}  \end{bmatrix}$ , $\dots$ , $\begin{bmatrix} 0  \\ 0 \\ \frac{(M-1)\lambda}{2}\end{bmatrix}$

此时由于阵元的相对位置中, 有两个维度的距离都是0，使得到达矢量的表达式可以大大简化。

$$
\boldsymbol{a}(\omega_{n})=
\begin{bmatrix}  1\\
\exp({j2\pi \frac{\langle \boldsymbol{r}_2', \boldsymbol{e}_{n} \rangle} {\lambda}}) \\
\vdots \\
\exp({j2\pi \frac{\langle \boldsymbol{r}_M', \boldsymbol{e}_{n} \rangle} {\lambda}}) 
\end{bmatrix}_{M \times 1} 
=
\begin{bmatrix}
    1 \\
    \exp(j \pi \sin(\phi)) \\
    \vdots \\
    \exp(j (M-1) \pi \sin(\phi))
\end{bmatrix}_{M \times 1}
$$


# matlab 函数
## matlab自带函数

matlab自带函数 ```steervec()``` 用于生成波束形成的方向向量。这个函数内部调用的是'Phased Array System Toolbox'中的内容，假如安装matlab时没有安装，可以在‘附加功能管理器’中重新安装。
>[matlab帮助文档](https://ww2.mathworks.cn/help/phased/ref/steervec.html)

该函数的基本用法如下

```matlab
sv = steervec(pos,ang)
```
 
需要注意的是： ```pos``` 中保存的阵元位置，**是以信号波长为单位的**，而不是 米(m)

## 自写的函数
[steering_vector.m](https://github.com/Huffer342-WSH/myNotes/blob/main/radar/project/steering_vector.m)
> 目前只支持三维空间坐标系的输入

```matlab
function sv = steering_vector(pos, ang, lambda)
    %steering_vector 计算导向矢量
    %   ANG 表示输入信号的方向。ANG 可以是 1xM 向量或 2xM 矩阵，其中 M 是输入信号的数量。
    %   如果 ANG 是一个 2xM 矩阵，每列以 [方位角; 俯仰角] 的形式（以度为单位）指定空间中的方向。
    %   方位角必须在 -180° 到 180° 之间，俯仰角必须在 -90° 到 90° 之间。
    %   方位角定义在 xy 平面内；它是与 x 轴（也是阵列法线方向）的夹角， y 轴正方向对应 90°。
    %   俯仰角定义为与 xy 平面的夹角
    %
    %   % Example:
    %   lambda = 1.25e-2;
    %   pos = [[0; 1; 0], [0; 5; 0], [0; 2; 0], [0; 3; 0]] * 0.5 * lambda;
    %   ang = [[10; 0], [20; 10], [30; 20], [0; 30]];
    %   my_sv = steering_vector(pos, ang, lambda);

    % 方位角
    azimuth = ang(1, :);
    % 俯仰角
    elevation = ang(2, :);
    % 目标方向的 单位方向向量
    unit_direction_vector = [cosd(elevation) .* cosd(azimuth); cosd(elevation) .* sind(azimuth); sind(elevation)];
    % 信号到达不同阵元的距离差，参考阵元坐标(0;0;0)
    distance = transpose(pos) * unit_direction_vector;
    % 计算导向矢量
    sv = exp(1j * 2 * pi * distance / lambda);
end

```


## 使用案例
[steering_vector_demo.m](https://github.com/Huffer342-WSH/myNotes/blob/main/radar/project/steering_vector_demo.m)

```matlab
clc;
%% 设置参数
lambda = 1.25e-2;
pos = [[0; 1; 0], [0; 5; 0], [0; 2; 0], [0; 3; 0]] * 0.5 * lambda;
ang = [[10; 0], [20; 10], [30; 20], [0; 30]];

%% 分别调用自写的函数和matlab的函数
my_sv = steering_vector(pos, ang, lambda);
matlab_sv = steervec(pos / lambda, ang);

%% 比较结果
matlab_sv
my_sv
diff = mean(abs((my_sv - matlab_sv) ./ matlab_sv), 'all')

```

```
matlab_sv =

   0.8549 + 0.5189i   0.4905 + 0.8715i   0.0946 + 0.9955i   1.0000 + 0.0000i
  -0.9155 + 0.4022i   0.5467 - 0.8373i   0.4561 + 0.8899i   1.0000 + 0.0000i
   0.4615 + 0.8871i  -0.5189 + 0.8549i  -0.9821 + 0.1883i   1.0000 + 0.0000i
  -0.0658 + 0.9978i  -0.9995 - 0.0329i  -0.2804 - 0.9599i   1.0000 + 0.0000i


my_sv =

   0.8549 + 0.5189i   0.4905 + 0.8715i   0.0946 + 0.9955i   1.0000 + 0.0000i
  -0.9155 + 0.4022i   0.5467 - 0.8373i   0.4561 + 0.8899i   1.0000 + 0.0000i
   0.4615 + 0.8871i  -0.5189 + 0.8549i  -0.9821 + 0.1883i   1.0000 + 0.0000i
  -0.0658 + 0.9978i  -0.9995 - 0.0329i  -0.2804 - 0.9599i   1.0000 + 0.0000i


diff =

   8.1272e-17
```
