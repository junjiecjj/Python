# 旁视阵列全维 STAP 数学原理与代码对应说明

本文档严格对应 MATLAB 程序 `STAP_Ward_Guerci_修正版_全中文.m` 的数学模型与处理流程。代码采用全维空时自适应处理（Full-Dimension STAP），在一个由 $N$ 个阵元和 $M$ 个相干脉冲构成的 $MN$ 维联合空时空间中，对目标、地杂波、点状干扰机和白噪声进行统一建模与处理。

需要特别注意，本代码采用的空时数据排列顺序为“脉冲维在外、阵元维在内”，因此所有空时导向矢量统一写成

$$
\mathbf v(f_s,f_d)
=
\mathbf b(f_d)\otimes\mathbf a(f_s).
$$

这一 Kronecker 积顺序必须在目标、杂波、干扰机、谱扫描和 STAP 权重响应中保持一致。

---

## 一、基本参数与归一化变量

代码设置均匀线阵阵元数为 $N$，一个相干处理区间（CPI）包含 $M$ 个脉冲，因此联合空时维数为

$$
D=MN.
$$

工作波长为 $\lambda$，阵元间距为 $d$。代码采用半波长阵元间距

$$
d=\frac{\lambda}{2}.
$$

噪声功率记为 $\sigma_n^2$，代码中为 `noisePower`。根据 CNR、SNR 和 JNR，可分别得到杂波、目标和干扰机功率

$$
P_c
=
\sigma_n^2 10^{\mathrm{CNR}/10},
$$

$$
P_t
=
\sigma_n^2 10^{\mathrm{SNR}/10},
$$

$$
P_j
=
\sigma_n^2 10^{\mathrm{JNR}/10}.
$$

代码中对应为

```matlab
clutterPower = noisePower * 10^(CNR/10);
tgtPower     = noisePower * 10^(SNR/10);
jammerPower  = noisePower * 10^(JNR/10);
```

### 1. 归一化空间频率

对于到达角为 $\theta$ 的窄带平面波，定义归一化空间频率

$$
f_s
=
\frac{d}{\lambda}\sin\theta.
$$

当 $d=\lambda/2$ 时，

$$
f_s
=
\frac{1}{2}\sin\theta,
$$

因此对于 $\theta\in[-90^\circ,90^\circ]$，

$$
f_s\in[-0.5,0.5].
$$

### 2. 归一化多普勒频率

定义归一化多普勒频率为

$$
f_d
=
\frac{f_D}{\mathrm{PRF}}
=
f_D T_r,
$$

其中 $f_D$ 为物理多普勒频率，$\mathrm{PRF}$ 为脉冲重复频率，$T_r=1/\mathrm{PRF}$ 为脉冲重复周期。

通常只考察一个无模糊多普勒区间

$$
f_d\in[-0.5,0.5].
$$

---

## 二、空间导向矢量与多普勒导向矢量

### 1. 空间导向矢量

对于归一化空间频率 $f_s$，代码采用

$$
\mathbf a(f_s)
=
\begin{bmatrix}
1 &
e^{-j2\pi f_s} &
\cdots &
e^{-j2\pi(N-1)f_s}
\end{bmatrix}^{T}.
$$

即

$$
\mathbf a(f_s)
\in
\mathbb C^{N\times1}.
$$

对应代码为

```matlab
a = exp(-1j*2*pi*f_s*[0:N-1].');
```

### 2. 慢时间多普勒导向矢量

对于归一化多普勒频率 $f_d$，慢时间导向矢量为

$$
\mathbf b(f_d)
=
\begin{bmatrix}
1 &
e^{-j2\pi f_d} &
\cdots &
e^{-j2\pi(M-1)f_d}
\end{bmatrix}^{T},
$$

其中

$$
\mathbf b(f_d)
\in
\mathbb C^{M\times1}.
$$

对应代码为

```matlab
b = exp(-1j*2*pi*f_d*[0:M-1].');
```

---

## 三、空时联合导向矢量

本代码使用

$$
\boxed{
\mathbf v(f_s,f_d)
=
\mathbf b(f_d)\otimes\mathbf a(f_s)
}
$$

作为一个角度—多普勒单元的空时导向矢量。

因为

$$
\mathbf b(f_d)\in\mathbb C^{M\times1},
\qquad
\mathbf a(f_s)\in\mathbb C^{N\times1},
$$

所以

$$
\mathbf v(f_s,f_d)
\in
\mathbb C^{MN\times1}.
$$

展开后

$$
\mathbf v(f_s,f_d)
=
\begin{bmatrix}
b_1\mathbf a\\
b_2\mathbf a\\
\vdots\\
b_M\mathbf a
\end{bmatrix}.
$$

这对应于每个脉冲包含 $N$ 个阵元数据，并按照脉冲依次堆叠的空时数据组织方式。

因此，一个 CPI 内的接收数据可以统一写成

$$
\mathbf x
\in
\mathbb C^{MN\times1}.
$$

---

# 四、杂波协方差矩阵 $R_c$

## 4.1 旁视阵列中的杂波脊

对于旁视机载雷达，静止地面杂波的空间频率与归一化多普勒之间具有确定的耦合关系。代码写成

$$
\boxed{
f_{d,c}
=
\beta f_s
}
$$

其中 $\beta$ 为杂波脊斜率参数。

代码当前取

$$
\beta=1,
$$

因此杂波在归一化空间频率—归一化多普勒平面中位于

$$
f_{d,c}=f_s
$$

这条直线上。

需要强调的是，$\beta$ 只用于描述杂波脊的位置关系，并不限制整个二维 Doppler 扫描范围。因此修正版代码将二维 Doppler 扫描轴独立设置为

$$
f_d\in[-0.5,0.5].
$$

---

## 4.2 杂波块离散模型

代码将连续地面杂波离散为 $K$ 个独立杂波块。第 $k$ 个杂波块对应空间频率

$$
f_{s,k}
$$

以及由杂波脊确定的归一化多普勒

$$
f_{d,c,k}
=
\beta f_{s,k}.
$$

其空间导向矢量和慢时间导向矢量分别为

$$
\mathbf a_{c,k}
=
\mathbf a(f_{s,k}),
$$

$$
\mathbf b_{c,k}
=
\mathbf b(f_{d,c,k}).
$$

代码将杂波功率直接包含在空时导向矢量中：

$$
\boxed{
\mathbf v_{c,k}
=
\sqrt{P_c}
\left(
\mathbf b_{c,k}
\otimes
\mathbf a_{c,k}
\right).
}
$$

对应

```matlab
v_clutter = sqrt(clutterPower) * ...
            kron(b_clutter, a_clutter);
```

---

## 4.3 杂波协方差矩阵

若不同杂波块相互不相关，则总杂波协方差可表示为各杂波块外积的叠加。

代码采用

$$
\boxed{
\mathbf R_c
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf v_{c,k}\mathbf v_{c,k}^{H}.
}
$$

由于 $\mathbf v_{c,k}$ 中已经包含 $\sqrt{P_c}$，因此也可以写成

$$
\mathbf R_c
=
\frac{P_c}{K}
\sum_{k=1}^{K}
\mathbf s_{c,k}\mathbf s_{c,k}^{H},
$$

其中

$$
\mathbf s_{c,k}
=
\mathbf b_{c,k}\otimes\mathbf a_{c,k}
$$

是不包含功率系数的单位空时导向矢量。

定义杂波空时导向矩阵

$$
\mathbf V_c
=
\begin{bmatrix}
\mathbf v_{c,1} &
\mathbf v_{c,2} &
\cdots &
\mathbf v_{c,K}
\end{bmatrix},
$$

则有

$$
\boxed{
\mathbf R_c
=
\frac{1}{K}
\mathbf V_c\mathbf V_c^{H}.
}
$$

这与程序循环累加 `v_clutter*v_clutter'` 后再除以杂波块数完全一致。

---

# 五、噪声协方差矩阵 $R_n$

代码假设不同阵元和不同脉冲上的接收机热噪声相互独立，并且具有相同功率 $\sigma_n^2$。

因此空时白噪声协方差矩阵为

$$
\boxed{
\mathbf R_n
=
\sigma_n^2\mathbf I_{MN}.
}
$$

其中 $\mathbf I_{MN}$ 为 $MN\times MN$ 单位矩阵。

对应代码为

```matlab
Rn = noisePower * eye(M*N);
```

---

# 六、目标模型与目标协方差矩阵 $R_t$

代码中的目标具有确定的方位角 $\theta_t$ 和确定的归一化多普勒 $f_{d,t}$。

目标归一化空间频率为

$$
f_{s,t}
=
\frac{d}{\lambda}\sin\theta_t.
$$

目标空间和慢时间导向矢量分别为

$$
\mathbf a_t
=
\mathbf a(f_{s,t}),
$$

$$
\mathbf b_t
=
\mathbf b(f_{d,t}).
$$

因此目标空时导向矢量为

$$
\boxed{
\mathbf v_t
=
\sqrt{P_t}
\left(
\mathbf b_t\otimes\mathbf a_t
\right).
}
$$

代码中当前参数为

$$
\theta_t=0^\circ,
$$

$$
f_{d,t}=-0.1.
$$

由于代码将目标建模为单个确定角度—多普勒单元，因此其协方差矩阵为秩 1 矩阵

$$
\boxed{
\mathbf R_t
=
\mathbf v_t\mathbf v_t^{H}.
}
$$

---

# 七、点状干扰机模型与 $R_j$

当前程序采用的不是慢时间白的宽带压制式干扰，而是一个具有确定空间方向和确定多普勒的点状空时干扰机。

干扰机方位角为 $\theta_j$，对应归一化空间频率

$$
f_{s,j}
=
\frac{d}{\lambda}\sin\theta_j.
$$

其归一化多普勒为 $f_{d,j}$。

对应导向矢量为

$$
\mathbf a_j
=
\mathbf a(f_{s,j}),
$$

$$
\mathbf b_j
=
\mathbf b(f_{d,j}).
$$

加入干扰功率后

$$
\boxed{
\mathbf v_j
=
\sqrt{P_j}
\left(
\mathbf b_j\otimes\mathbf a_j
\right).
}
$$

因此当前程序中的干扰协方差矩阵为

$$
\boxed{
\mathbf R_j
=
\mathbf v_j\mathbf v_j^{H}.
}
$$

它也是秩 1 协方差矩阵。

当前代码设置

$$
\theta_j=-30^\circ,
$$

$$
f_{d,j}=0.1.
$$

因此该干扰在二维空间频率—多普勒平面中主要表现为一个局部强干扰点，而不是沿整个 Doppler 方向展开的宽带干扰条带。

---

# 八、总干扰加噪声协方差矩阵及其特征值谱

用于设计 STAP 权重的干扰加噪声协方差矩阵为

$$
\boxed{
\mathbf R
=
\mathbf R_c
+
\mathbf R_j
+
\mathbf R_n.
}
$$

注意这里不包含目标协方差 $\mathbf R_t$。

其物理意义是描述目标之外的所有干扰与噪声在 $MN$ 维空时空间中的二阶统计结构。

代码对 $\mathbf R$ 做奇异值分解：

$$
\mathbf R
=
\mathbf U\mathbf \Sigma\mathbf V^{H}.
$$

由于 $\mathbf R$ 是 Hermitian 正定矩阵，理论上其奇异值等于特征值，因此图中的 `diag(S)` 可以用于观察干扰加噪声协方差的特征值谱。

如果写成特征值分解，

$$
\mathbf R
=
\mathbf U
\mathbf \Lambda
\mathbf U^{H},
$$

则

$$
\mathbf \Lambda
=
\operatorname{diag}
\left(
\lambda_1,\lambda_2,\ldots,\lambda_{MN}
\right).
$$

强杂波与强干扰通常会使少数特征方向具有较大的特征值，而纯白噪声对应的特征值主要分布在噪声底附近。

因此特征值谱能够直观反映干扰子空间与噪声子空间之间的能量差异。

---

# 九、STAP 处理前的总接收协方差

为了绘制处理前的目标、杂波、干扰机和噪声整体二维功率谱，代码另外构造

$$
\boxed{
\mathbf R_{\mathrm{total}}
=
\mathbf R_c
+
\mathbf R_j
+
\mathbf R_n
+
\mathbf R_t.
}
$$

这里与上一节的 $\mathbf R$ 必须严格区分：

$$
\mathbf R
=
\mathbf R_c+\mathbf R_j+\mathbf R_n
$$

用于设计最优 STAP 权重；

而

$$
\mathbf R_{\mathrm{total}}
=
\mathbf R+\mathbf R_t
$$

用于观察包含目标在内的总接收数据空间功率分布。

---

# 十、常规 DBF / Bartlett 空时功率谱

对于任意测试点 $(f_s,f_d)$，首先构造测试空时导向矢量

$$
\mathbf v
=
\mathbf v(f_s,f_d)
=
\mathbf b(f_d)\otimes\mathbf a(f_s).
$$

代码中的常规 DBF/Bartlett 功率谱定义为

$$
\boxed{
P_{\mathrm{DBF}}(f_s,f_d)
=
\mathbf v^{H}
\mathbf R_{\mathrm{total}}
\mathbf v.
}
$$

对应代码

```matlab
P_dbf(j_fdNormalized, iSpatialFreq) = ...
    v' * R_total * v;
```

它可以理解为使用测试导向矢量 $\mathbf v$ 对总接收协方差进行匹配扫描。

当测试点与某个强目标、杂波块或干扰机的角度—多普勒位置一致时，$\mathbf v$ 会与该分量具有较大的相关性，因此 $P_{\mathrm{DBF}}$ 会出现较大值。

DBF/Bartlett 谱的特点是实现简单，但空间—多普勒分辨率和旁瓣抑制能力受传统匹配波束宽度限制。

---

# 十一、Capon / MVDR 空时功率谱

Capon 谱利用总接收协方差矩阵的逆矩阵进行自适应谱估计。

对于每一个测试空时导向矢量 $\mathbf v$，考虑无失真约束

$$
\mathbf w^H\mathbf v=1,
$$

并最小化输出总功率

$$
\min_{\mathbf w}
\mathbf w^H
\mathbf R_{\mathrm{total}}
\mathbf w.
$$

其最优权重为

$$
\mathbf w_{\mathrm C}
=
\frac{
\mathbf R_{\mathrm{total}}^{-1}\mathbf v
}{
\mathbf v^H
\mathbf R_{\mathrm{total}}^{-1}
\mathbf v
}.
$$

将其代回最小输出功率，可以得到 Capon/MVDR 空时谱

$$
\boxed{
P_{\mathrm{Capon}}(f_s,f_d)
=
\frac{
1
}{
\mathbf v^H
\mathbf R_{\mathrm{total}}^{-1}
\mathbf v
}.
}
$$

代码没有显式计算 $\mathbf R_{\mathrm{total}}^{-1}$，而采用

```matlab
R_total \ v
```

计算

$$
\mathbf R_{\mathrm{total}}^{-1}\mathbf v.
$$

因此实现为

```matlab
P_capon = 1./real(v'*(R_total\v));
```

这在数学上仍然严格对应

$$
\frac{1}
{\mathbf v^H\mathbf R_{\mathrm{total}}^{-1}\mathbf v}.
$$

与普通 DBF 谱相比，Capon 谱利用了接收数据的协方差结构，因此通常具有更高的角度—多普勒分辨能力。

---

# 十二、最优 STAP 权重

## 12.1 空时接收模型

在目标待检测距离单元中，可以写成

$$
\mathbf x
=
\alpha_t\mathbf s_t
+
\mathbf q,
$$

其中 $\mathbf s_t$ 为不包含目标随机复幅度的目标空时导向矢量，$\mathbf q$ 包含杂波、干扰机和噪声。

定义

$$
E[\mathbf q\mathbf q^H]
=
\mathbf R.
$$

对一个空时权重 $\mathbf w$，滤波器输出为

$$
y
=
\mathbf w^H\mathbf x.
$$

若目标功率为 $P_t$，则输出目标功率为

$$
P_{t,\mathrm{out}}
=
P_t
\left|
\mathbf w^H\mathbf s_t
\right|^2.
$$

输出干扰加噪声功率为

$$
P_{i+n,\mathrm{out}}
=
\mathbf w^H
\mathbf R
\mathbf w.
$$

所以输出 SINR 为

$$
\boxed{
\mathrm{SINR}_{\mathrm{out}}
=
\frac{
P_t
\left|
\mathbf w^H\mathbf s_t
\right|^2
}{
\mathbf w^H
\mathbf R
\mathbf w
}.
}
$$

---

## 12.2 最大输出 SINR 解

最大化上式可得最优权重方向

$$
\boxed{
\mathbf w_{\mathrm{opt}}
\propto
\mathbf R^{-1}\mathbf s_t.
}
$$

由于程序中的 `v_tgt` 已经包含 $\sqrt{P_t}$，乘上这一非零常数不会改变最优权重方向，所以代码直接使用

$$
\boxed{
\mathbf w_{\mathrm{opt}}
=
\mathbf R^{-1}\mathbf v_t.
}
$$

对应 MATLAB 实现

```matlab
wopt = R \ v_tgt;
```

这里使用 `R\v_tgt` 而不是 `inv(R)*v_tgt`，数学含义相同，但数值实现更稳定。

---

## 12.3 MVDR 归一化形式

最大 SINR 解只确定权重方向，任意非零比例系数都不会改变输出 SINR。

如果进一步要求目标方向满足无失真约束

$$
\mathbf w^H\mathbf v_t=1,
$$

则可写成

$$
\boxed{
\mathbf w_{\mathrm{MVDR}}
=
\frac{
\mathbf R^{-1}\mathbf v_t
}{
\mathbf v_t^H
\mathbf R^{-1}\mathbf v_t
}.
}
$$

代码中保留了这一归一化形式作为注释。

因此未归一化的 $\mathbf R^{-1}\mathbf v_t$ 和归一化 MVDR 形式在最大 SINR 意义下是等价的，只是输出幅度尺度不同。

---

# 十三、为什么 $R^{-1}$ 能够抑制杂波和干扰

令干扰加噪声协方差矩阵的特征值分解为

$$
\mathbf R
=
\mathbf U
\mathbf \Lambda
\mathbf U^H,
$$

其中

$$
\mathbf \Lambda
=
\operatorname{diag}
(\lambda_1,\ldots,\lambda_{MN}).
$$

则

$$
\mathbf R^{-1}
=
\mathbf U
\mathbf \Lambda^{-1}
\mathbf U^H,
$$

即

$$
\mathbf R^{-1}
=
\mathbf U
\operatorname{diag}
\left(
\frac{1}{\lambda_1},
\ldots,
\frac{1}{\lambda_{MN}}
\right)
\mathbf U^H.
$$

如果某个空时特征方向包含强杂波或强干扰，则其 $\lambda_i$ 很大，在 $\mathbf R^{-1}$ 中就会被乘以较小的 $1/\lambda_i$。

因此

$$
\mathbf R^{-1}\mathbf v_t
$$

会自动降低目标 steering vector 在高干扰子空间中的分量，从而在保持目标响应的同时对杂波和干扰形成自适应抑制。

这就是 STAP 联合利用空间维和慢时间维统计信息的核心。

---

# 十四、STAP 最优权重的二维空时响应

得到 $\mathbf w_{\mathrm{opt}}$ 后，代码并不是重新生成随机接收数据做检测，而是扫描整个 $(f_s,f_d)$ 平面，绘制最优权重的二维空时响应。

对于任意测试点

$$
(f_s,f_d),
$$

构造

$$
\mathbf v(f_s,f_d)
=
\mathbf b(f_d)\otimes\mathbf a(f_s).
$$

最优权重在该点的复响应为

$$
\boxed{
Y(f_s,f_d)
=
\mathbf w_{\mathrm{opt}}^H
\mathbf v(f_s,f_d).
}
$$

绘图时使用功率形式

$$
\boxed{
P_Y(f_s,f_d)
=
\left|
Y(f_s,f_d)
\right|^2.
}
$$

转换为 dB 后为

$$
P_{Y,\mathrm{dB}}
=
10\log_{10}
\left|
Y(f_s,f_d)
\right|^2.
$$

因此代码第 (8) 部分画出的图应理解为

> **最优 STAP 空时滤波器的二维方向响应**

而不是直接的“目标检测结果”。

由于 $\mathbf w_{\mathrm{opt}}$ 是根据 $\mathbf R_c+\mathbf R_j+\mathbf R_n$ 设计的，因此该响应会在强杂波和强干扰对应的空时子空间附近产生明显抑制。

---

# 十五、SINR Loss 的数学原理

代码最后计算整个二维角度—多普勒平面上的 SINR Loss。

对于任意测试 steering vector $\mathbf v$，若目标功率归一化为 $P_t=1$，最大输出 SINR 为

$$
\boxed{
\mathrm{SINR}_{\max}
=
\mathbf v^H
\mathbf R^{-1}
\mathbf v.
}
$$

这是因为最优权重满足

$$
\mathbf w_{\mathrm{opt}}
\propto
\mathbf R^{-1}\mathbf v.
$$

代码对应

```matlab
SINR_current = real(v'*(R\v));
```

---

## 15.1 白噪声条件下的最优 SNR

如果不存在杂波和干扰，只有功率为 $\sigma_n^2$ 的白噪声，则

$$
\mathbf R_n
=
\sigma_n^2\mathbf I.
$$

于是

$$
\mathbf R_n^{-1}
=
\frac{1}{\sigma_n^2}\mathbf I.
$$

白噪声背景下的最优输出 SNR 为

$$
\boxed{
\mathrm{SNR}_{\mathrm{opt}}
=
\frac{
\mathbf v^H\mathbf v
}{
\sigma_n^2
}.
}
$$

代码对应

```matlab
SNRopt = real(v'*v)/noisePower;
```

---

## 15.2 SINR Loss

定义 SINR Loss 为实际杂波、干扰和噪声环境下的最大输出 SINR，相对于纯白噪声条件下最优输出 SNR 的比值：

$$
\boxed{
L_{\mathrm{SINR}}
=
\frac{
\mathrm{SINR}_{\max}
}{
\mathrm{SNR}_{\mathrm{opt}}
}.
}
$$

代入前面的表达式可得

$$
\boxed{
L_{\mathrm{SINR}}
=
\frac{
\mathbf v^H
\mathbf R^{-1}
\mathbf v
}{
\mathbf v^H\mathbf v/\sigma_n^2
}.
}
$$

等价地，

$$
L_{\mathrm{SINR}}
=
\sigma_n^2
\frac{
\mathbf v^H
\mathbf R^{-1}
\mathbf v
}{
\mathbf v^H\mathbf v
}.
$$

通常有

$$
0<L_{\mathrm{SINR}}\le1.
$$

转换成 dB：

$$
\boxed{
L_{\mathrm{SINR,dB}}
=
10\log_{10}
L_{\mathrm{SINR}}.
}
$$

因此理想白噪声条件对应

$$
L_{\mathrm{SINR,dB}}=0\ \mathrm{dB},
$$

而杂波和干扰导致的性能损失表现为负 dB。

代码最后固定

$$
f_s=0
$$

的空间频率切片，并沿 Doppler 轴绘制

$$
L_{\mathrm{SINR}}(0,f_d).
$$

这能够直观看到杂波脊及干扰附近的 SINR 损失。

---

# 十六、代码中各协方差矩阵的区别

程序中出现了两个非常容易混淆的协方差矩阵。

### 1. STAP 权重设计使用

$$
\boxed{
\mathbf R
=
\mathbf R_c
+
\mathbf R_j
+
\mathbf R_n.
}
$$

它只包含干扰与噪声，不包含目标。

对应代码：

```matlab
R = Rc + Rj + Rn;
```

并用于

```matlab
wopt = R \ v_tgt;
```

即

$$
\mathbf w_{\mathrm{opt}}
=
\mathbf R^{-1}\mathbf v_t.
$$

### 2. 处理前谱估计使用

$$
\boxed{
\mathbf R_{\mathrm{total}}
=
\mathbf R_c
+
\mathbf R_j
+
\mathbf R_n
+
\mathbf R_t.
}
$$

它包含目标，因此用于观察总接收数据中的目标、杂波和干扰分布。

对应代码：

```matlab
R_total = Rc + Rj + Rn + Rt;
```

并用于 DBF 和 Capon 谱。

这两个矩阵的用途不能互换。

---

# 十七、程序完整数学处理链

整份代码可以概括为如下链条。

首先，根据角度和 Doppler 构造空间、慢时间和空时导向矢量：

$$
\theta
\rightarrow
f_s
\rightarrow
\mathbf a(f_s),
$$

$$
f_d
\rightarrow
\mathbf b(f_d),
$$

$$
(\mathbf a,\mathbf b)
\rightarrow
\mathbf v
=
\mathbf b\otimes\mathbf a.
$$

然后建立四类二阶统计量：

$$
\mathbf R_c
=
\frac{1}{K}
\sum_k
\mathbf v_{c,k}\mathbf v_{c,k}^H,
$$

$$
\mathbf R_n
=
\sigma_n^2\mathbf I,
$$

$$
\mathbf R_t
=
\mathbf v_t\mathbf v_t^H,
$$

$$
\mathbf R_j
=
\mathbf v_j\mathbf v_j^H.
$$

接着形成

$$
\mathbf R
=
\mathbf R_c+\mathbf R_j+\mathbf R_n
$$

和

$$
\mathbf R_{\mathrm{total}}
=
\mathbf R+\mathbf R_t.
$$

利用 $\mathbf R_{\mathrm{total}}$ 得到处理前二维谱：

$$
P_{\mathrm{DBF}}
=
\mathbf v^H
\mathbf R_{\mathrm{total}}
\mathbf v,
$$

$$
P_{\mathrm{Capon}}
=
\frac{1}{
\mathbf v^H
\mathbf R_{\mathrm{total}}^{-1}
\mathbf v
}.
$$

利用 $\mathbf R$ 和目标 steering vector 得到全维 STAP 权重：

$$
\boxed{
\mathbf w_{\mathrm{opt}}
=
\mathbf R^{-1}\mathbf v_t.
}
$$

再扫描整个角度—多普勒平面：

$$
Y(f_s,f_d)
=
\mathbf w_{\mathrm{opt}}^H
\mathbf v(f_s,f_d).
$$

最后计算性能损失：

$$
L_{\mathrm{SINR}}
=
\frac{
\mathbf v^H
\mathbf R^{-1}
\mathbf v
}{
\mathbf v^H\mathbf v/\sigma_n^2
}.
$$

因此整份程序的逻辑可以总结为

$$
\boxed{
\text{空时建模}
\rightarrow
\text{协方差建模}
\rightarrow
\text{处理前二维谱}
\rightarrow
\text{最优 STAP 权重}
\rightarrow
\text{二维空时响应}
\rightarrow
\text{SINR Loss}.
}
$$

---

# 十八、代码与公式对应表

| MATLAB 变量 | 数学含义 |
|---|---|
| `SpatialFreqGrid_normalized` | 归一化空间频率 $f_s$ |
| `fdGrid_Normalized` | 归一化多普勒 $f_d$ |
| `a_clutter` | 杂波空间导向矢量 $\mathbf a_{c,k}$ |
| `b_clutter` | 杂波 Doppler 导向矢量 $\mathbf b_{c,k}$ |
| `v_clutter` | 杂波空时导向矢量 $\mathbf v_{c,k}$ |
| `Rc` | 杂波协方差 $\mathbf R_c$ |
| `Rn` | 白噪声协方差 $\mathbf R_n$ |
| `a_tgt` | 目标空间导向矢量 $\mathbf a_t$ |
| `b_tgt` | 目标 Doppler 导向矢量 $\mathbf b_t$ |
| `v_tgt` | 目标空时导向矢量 $\mathbf v_t$ |
| `Rt` | 目标协方差 $\mathbf R_t$ |
| `a_jammer` | 干扰机空间导向矢量 $\mathbf a_j$ |
| `b_jammer` | 干扰机 Doppler 导向矢量 $\mathbf b_j$ |
| `v_jammer` | 干扰机空时导向矢量 $\mathbf v_j$ |
| `Rj` | 干扰机协方差 $\mathbf R_j$ |
| `R` | 干扰加噪声协方差 $\mathbf R_c+\mathbf R_j+\mathbf R_n$ |
| `R_total` | 含目标的总接收协方差 $\mathbf R+\mathbf R_t$ |
| `P_dbf` | DBF/Bartlett 空时谱 |
| `P_capon` | Capon/MVDR 空时谱 |
| `wopt` | 最大输出 SINR 的 STAP 权重 |
| `Y` | STAP 权重二维空时响应 |
| `SINR_loss` | 二维 SINR Loss |

---

# 十九、当前代码所对应的理想化假设

为了使数学原理与代码一一对应，当前仿真采用以下理想化条件：

1. 阵列为均匀线阵；
2. 每个 CPI 内阵列和目标运动参数不变；
3. 地面杂波由多个相互独立的离散杂波块构成；
4. 杂波块位于理想旁视杂波脊 $f_{d,c}=\beta f_s$ 上；
5. 噪声在阵元维和脉冲维上均为空间—时间白噪声；
6. 目标为单个确定角度、确定 Doppler 的秩 1 信号；
7. 干扰机为单个确定角度、确定 Doppler 的秩 1 点状干扰；
8. $\mathbf R_c$、$\mathbf R_j$ 和 $\mathbf R_n$ 直接由理论模型获得，而不是通过有限训练快拍估计。

因此这份代码属于理想理论协方差条件下的 **full-dimension STAP 原理验证**。实际 STAP 通常还需要进一步考虑训练样本有限、协方差估计误差、非均匀杂波、阵列误差以及降维处理等问题。
