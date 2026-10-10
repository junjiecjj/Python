谢老师，贸然打扰，想请教您下面的一些问题，您有空的时候看看即可，十分感谢：

我在看文献中经常看到不行形式的MIMO雷达方程，有的文献其实没有交代清楚，或者说本身就搞不清楚这些模型之间明晰的区别，只是随手拿了一个来用用。 我问过其他人为啥有这些区别，有的说只是为了数学形式上方便，但是我觉得是有物理区别的。这些区别会影响问题的建模和后续的分析；

(1) 文献中很多关于雷达的分类，比如：monostatic、bistatic、statistic、coloated，这些分类的区别是什么？会对问题建模和求解造成什么影响？

(2) 为啥有的文献喜欢对雷达方程进行线性化，有些不线性化，直接以矩阵的形式处理？本质原因是什么？和上面的雷达分类有关系吗？

我考虑的划分如下图，但是我总感觉也不完全对：我好像把雷达方向是否线性化与不同雷达种类对应的雷达方程混淆了，换句话说 这是两个不同层面的问题，雷达的划分可以体现在雷达方程的参数形式上，而不是是否线性化上。







# 空时自适应处理（STAP）原理

## 一、STAP 的处理对象

空时自适应处理（Space-Time Adaptive Processing, STAP）用于阵列雷达中的杂波与干扰抑制。其核心不是分别完成“空间波束形成”和“慢时间多普勒滤波”，而是把阵列维与脉冲维组成一个统一的空时数据向量，并直接在该联合空间中设计一个 $NM$ 维自适应权重。

设接收阵列包含 $N$ 个阵元，每个相干处理区间包含 $M$ 个脉冲。本文采用如下数据堆叠方式：

$$
\mathbf{x}
=
\begin{bmatrix}
\mathbf{x}_1\\
\mathbf{x}_2\\
\vdots\\
\mathbf{x}_N
\end{bmatrix}
\in\mathbb{C}^{NM\times 1},
$$

其中

$$
\mathbf{x}_n
=
[x_n(0),x_n(1),\ldots,x_n(M-1)]^T
$$

表示第 $n$ 个阵元在 $M$ 个脉冲上的慢时间采样。

在这个堆叠约定下，后续所有空时导向矢量统一写成

$$
\boxed{
\mathbf{s}(\theta,\nu)
=
\mathbf{a}(\theta)\otimes\mathbf{b}(\nu)
}
$$

其中 $\otimes$ 表示 Kronecker 积。

---

## 二、空间维导向矢量

考虑阵元间距为 $d$ 的 $N$ 元均匀线阵（ULA），工作波长为 $\lambda$。

定义归一化空间频率

$$
u(\theta)
=
\frac{d}{\lambda}\sin\theta.
$$

则空间导向矢量为

$$
\boxed{
\mathbf{a}(\theta)
=
\begin{bmatrix}
1\\
e^{j2\pi u(\theta)}\\
\vdots\\
e^{j2\pi(N-1)u(\theta)}
\end{bmatrix}
}
\in\mathbb{C}^{N\times1}.
$$

它描述同一个窄带平面波到达不同阵元时产生的确定性相位差。

MATLAB 代码中对应为

```matlab
elementIndex = (0:N-1).';
u = (d/lambda) * sind(thetaDeg);
a = exp(1j*2*pi*elementIndex*u);
```

这里必须区分“物理角度 $\theta$”与“归一化空间频率 $u$”。二者不是同一个变量。

---

## 三、慢时间多普勒导向矢量

设脉冲重复周期为 $T_r$，目标多普勒频率为 $f_d$。定义归一化多普勒频率

$$
\boxed{
\nu=f_dT_r
}
$$

单位为 cycles per PRI。

经过 $M$ 个相干脉冲后，目标在慢时间维上的相位呈等差递增，因此多普勒导向矢量为

$$
\boxed{
\mathbf{b}(\nu)
=
\begin{bmatrix}
1\\
e^{j2\pi\nu}\\
\vdots\\
e^{j2\pi(M-1)\nu}
\end{bmatrix}
}
\in\mathbb{C}^{M\times1}.
$$

MATLAB 代码中对应为

```matlab
pulseIndex = (0:M-1).';
b = exp(1j*2*pi*pulseIndex*nu);
```

当采用归一化多普勒描述时，未模糊区间通常写成

$$
-\frac{1}{2}\le\nu\le\frac{1}{2}.
$$

---

## 四、空时联合处理原理（STAP 核心）

### 4.1 空时联合导向矢量

一个位于角度 $\theta$、归一化多普勒 $\nu$ 的窄带目标，其空间结构由 $\mathbf a(\theta)$ 描述，慢时间结构由 $\mathbf b(\nu)$ 描述。

在本文的数据排列方式下，二者联合得到

$$
\boxed{
\mathbf{s}(\theta,\nu)
=
\mathbf{a}(\theta)\otimes\mathbf{b}(\nu)
}
\in\mathbb{C}^{NM\times1}.
$$

展开后为

$$
\mathbf{s}(\theta,\nu)
=
\begin{bmatrix}
a_1(\theta)\mathbf b(\nu)\\
a_2(\theta)\mathbf b(\nu)\\
\vdots\\
a_N(\theta)\mathbf b(\nu)
\end{bmatrix}.
$$

因此 STAP 不是分别求一个 $N$ 维空间权重和一个 $M$ 维多普勒权重，而是直接求

$$
\boxed{
\mathbf w\in\mathbb C^{NM\times1}
}
$$

的联合空时权重。

需要特别注意：

$$
\mathbf{s}(\theta,\nu)
=
\mathbf a(\theta)\otimes\mathbf b(\nu)
$$

并不意味着最优 STAP 权重也必须满足

$$
\mathbf w
=
\mathbf w_s\otimes\mathbf w_t.
$$

完整 STAP 允许 $\mathbf w$ 是一般的 $NM$ 维复向量，这正是联合空时自适应处理相对于可分离处理的关键区别。

---

### 4.2 空时接收模型

在一个待检测距离单元内，可将接收数据写成

$$
\boxed{
\mathbf x
=
\alpha_0\mathbf s_0
+
\mathbf c
+
\mathbf j
+
\mathbf n
}
$$

其中：

$$
\mathbf s_0
=
\mathbf s(\theta_0,\nu_0)
$$

为目标空时导向矢量；

$\alpha_0$ 为目标复幅度；

$\mathbf c$ 为杂波；

$\mathbf j$ 为主动干扰；

$\mathbf n$ 为接收机热噪声。

STAP 输出为

$$
\boxed{
y=\mathbf w^H\mathbf x.
}
$$

目标输出功率为

$$
P_{s,\mathrm{out}}
=
\sigma_s^2
\left|
\mathbf w^H\mathbf s_0
\right|^2.
$$

如果把杂波、干扰和噪声统一视为干扰加噪声项

$$
\mathbf q
=
\mathbf c+\mathbf j+\mathbf n,
$$

其协方差矩阵为

$$
\boxed{
\mathbf R
=
E[\mathbf q\mathbf q^H]
=
\mathbf R_c+\mathbf R_j+\mathbf R_n.
}
$$

则输出干扰加噪声功率为

$$
P_{q,\mathrm{out}}
=
\mathbf w^H\mathbf R\mathbf w.
$$

因此输出 SINR 为

$$
\boxed{
\mathrm{SINR}_{\mathrm{out}}
=
\frac{
\sigma_s^2
|
\mathbf w^H\mathbf s_0
|^2
}{
\mathbf w^H\mathbf R\mathbf w
}.
}
$$

这就是 STAP 权重设计的基本优化目标。

---

### 4.3 最大输出 SINR 权重

最大化

$$
\frac{
|
\mathbf w^H\mathbf s_0
|^2
}{
\mathbf w^H\mathbf R\mathbf w
}
$$

得到的最优权重方向为

$$
\boxed{
\mathbf w
\propto
\mathbf R^{-1}\mathbf s_0.
}
$$

如果进一步加入无失真约束

$$
\mathbf w^H\mathbf s_0=1,
$$

则可写成标准 MVDR/STAP 形式：

$$
\boxed{
\mathbf w_{\mathrm{STAP}}
=
\frac{
\mathbf R^{-1}\mathbf s_0
}{
\mathbf s_0^H
\mathbf R^{-1}
\mathbf s_0
}.
}
$$

其中分母只起归一化作用，不改变最大 SINR 权重的方向。

MATLAB 中不建议显式计算 `inv(R)`，而应通过线性方程

```matlab
x = R \ sTarget;
w = x / (sTarget' * x);
```

实现

$$
\mathbf x=\mathbf R^{-1}\mathbf s_0.
$$

这样数值稳定性和计算效率通常更好。

---

### 4.4 为什么 $R^{-1}$ 能抑制杂波和干扰

$\mathbf R$ 描述干扰加噪声在 $NM$ 维空时空间中的能量分布。

如果某些空时方向具有很强的杂波或干扰功率，则 $\mathbf R$ 在这些方向上的特征值较大。经过

$$
\mathbf R^{-1}
$$

后，这些高能量特征方向会被显著衰减，而低干扰方向被相对保留。

因此

$$
\mathbf R^{-1}\mathbf s_0
$$

可以理解为先依据干扰统计结构进行“空时白化”，再保持对目标导向矢量的匹配。

STAP 的本质不是简单地在几个预先给定的位置放置零陷，而是依据整个空时干扰协方差结构，自适应地重新分配 $NM$ 个联合权重。

---

## 五、杂波空时模型

### 5.1 旁视阵列下的杂波脊

考虑平台以速度 $v$ 运动、地面散射体静止的典型旁视阵列模型。

一个位于角度 $\theta$ 的静止杂波块产生的多普勒频率为

$$
f_{d,c}(\theta)
=
\frac{2v}{\lambda}\sin\theta.
$$

归一化后

$$
\nu_c(\theta)
=
f_{d,c}(\theta)T_r
=
\frac{2vT_r}{\lambda}\sin\theta.
$$

又因为

$$
u(\theta)
=
\frac{d}{\lambda}\sin\theta,
$$

所以

$$
\boxed{
\nu_c(\theta)
=
\beta u(\theta),
\qquad
\beta
=
\frac{2vT_r}{d}.
}
$$

这说明静止地面杂波不是均匀铺满整个角度-多普勒平面，而是集中在一条由平台运动几何关系决定的曲线上，即杂波脊。

代码中对应为

```matlab
beta = 2*v*Tr/d;

clutterSpatialFreq = ...
    (d/lambda) * sind(clutterAngleDeg);

clutterDoppler = ...
    beta * clutterSpatialFreq;
```

---

### 5.2 杂波块离散模型

将地面杂波离散成 $K$ 个相互独立的杂波块，第 $k$ 个杂波块具有角度 $\theta_k$ 和多普勒

$$
\nu_{c,k}
=
\nu_c(\theta_k).
$$

其空时导向矢量为

$$
\boxed{
\mathbf s_{c,k}
=
\mathbf a(\theta_k)
\otimes
\mathbf b(\nu_{c,k}).
}
$$

如果不同杂波块相互不相关，则杂波协方差矩阵为

$$
\boxed{
\mathbf R_c
=
\sum_{k=1}^{K}
\sigma_{c,k}^2
\mathbf s_{c,k}
\mathbf s_{c,k}^H.
}
$$

若各杂波块等功率，

$$
\sigma_{c,k}^2=\sigma_c^2,
$$

定义

$$
\mathbf V_c
=
[
\mathbf s_{c,1},
\mathbf s_{c,2},
\ldots,
\mathbf s_{c,K}
],
$$

则

$$
\boxed{
\mathbf R_c
=
\sigma_c^2
\mathbf V_c\mathbf V_c^H.
}
$$

代码严格按照这一公式构造 `Rc`。

---

## 六、宽带压制式干扰机模型

本文代码采用典型的宽带压制式干扰机模型。

第 $q$ 个干扰机位于角度 $\theta_{j,q}$，空间导向矢量为

$$
\mathbf a_{j,q}
=
\mathbf a(\theta_{j,q}).
$$

假设该干扰机在不同脉冲之间近似白，即慢时间协方差为

$$
\mathbf I_M.
$$

在本文使用的

$$
\mathbf a\otimes\mathbf b
$$

数据排列下，该干扰机的空时协方差为

$$
\boxed{
\mathbf R_{j,q}
=
\sigma_{j,q}^2
\left(
\mathbf a_{j,q}
\mathbf a_{j,q}^H
\right)
\otimes
\mathbf I_M.
}
$$

多个相互独立干扰机时

$$
\boxed{
\mathbf R_j
=
\sum_q
\mathbf R_{j,q}.
}
$$

这种干扰在角度上具有明确方向性，但由于慢时间近似白，会在该角度处覆盖较宽的多普勒范围。

代码中对应为

```matlab
Rj = Rj + jammerPower * kron(aJammer*aJammer', eye(M));
```

如果研究的是具有确定脉间相位关系的窄带相干干扰，则需要另外为其建立慢时间导向矢量，此时其模型会变为

$$
\mathbf s_j
=
\mathbf a(\theta_j)\otimes\mathbf b(\nu_j)
$$

以及

$$
\mathbf R_j
=
\sigma_j^2
\mathbf s_j\mathbf s_j^H.
$$

本文代码没有采用这种点状角度-多普勒干扰模型。

---

## 七、白噪声与总协方差矩阵

若接收机噪声在阵元和脉冲之间均为白噪声，则

$$
\boxed{
\mathbf R_n
=
\sigma_n^2
\mathbf I_{NM}.
}
$$

于是完整干扰加噪声协方差为

$$
\boxed{
\mathbf R
=
\mathbf R_n
+
\mathbf R_c
+
\mathbf R_j.
}
$$

本文代码分别考察四种情况：

$$
\mathbf R_1=\mathbf R_n,
$$

$$
\mathbf R_2=\mathbf R_n+\mathbf R_c,
$$

$$
\mathbf R_3=\mathbf R_n+\mathbf R_j,
$$

$$
\boxed{
\mathbf R_4
=
\mathbf R_n+\mathbf R_c+\mathbf R_j.
}
$$

这样可以分别观察杂波和干扰机对 STAP 空时响应的影响。

---

## 八、角度-多普勒二维响应

得到 STAP 权重 $\mathbf w$ 后，对于任意扫描点 $(\theta,\nu)$，构造

$$
\mathbf s(\theta,\nu)
=
\mathbf a(\theta)
\otimes
\mathbf b(\nu).
$$

二维空时响应定义为

$$
\boxed{
G(\theta,\nu)
=
\left|
\mathbf w^H
\mathbf s(\theta,\nu)
\right|.
}
$$

采用 MVDR 归一化

$$
\mathbf w^H\mathbf s_0=1
$$

时，目标位置

$$
(\theta_0,\nu_0)
$$

处的理论响应为

$$
G(\theta_0,\nu_0)=1,
$$

即

$$
20\log_{10}G(\theta_0,\nu_0)=0\ \mathrm{dB}.
$$

代码通过遍历角度扫描网格和归一化多普勒网格得到二维响应图。

其中：

- 仅有白噪声时，权重退化为空时匹配权重；
- 加入杂波后，响应会沿杂波脊附近形成自适应抑制；
- 加入宽带压制式干扰机后，会在其空间到达角附近形成宽多普勒零陷；
- 同时存在杂波和干扰机时，完整 STAP 权重会在一个 $NM$ 维联合空间中同时处理两类干扰。

---

## 九、空时 Chebyshev 加窗与第二幅图

代码的第二幅图比较**均匀空时匹配**和 **Chebyshev 空时加窗**的角度-多普勒二维响应。需要强调，这一部分不是 STAP 的协方差自适应处理，而是固定权重旁瓣控制，用来说明空时加窗对二维旁瓣的影响。

未加窗时，采用目标空时导向矢量作为匹配权重。归一化后为

$$
\mathbf w_{\mathrm{uniform}}
=
\frac{\mathbf s_0}
{\mathbf s_0^H\mathbf s_0}.
$$

分别定义空间维 Chebyshev 窗 $$\mathbf g_s\in\mathbb C^{N\times1}$$ 和慢时间维 Chebyshev 窗 $$\mathbf g_t\in\mathbb C^{M\times1}$$。按照本文 $$\mathbf s=\mathbf a\otimes\mathbf b$$ 的空时堆叠方式，联合空时窗为

$$
\mathbf g
=
\mathbf g_s\otimes\mathbf g_t.
$$

将该固定窗作用在目标空时导向矢量上，得到

$$
\widetilde{\mathbf s}_0
=
\mathbf s_0\odot\mathbf g,
$$

其中 $$\odot$$ 表示逐元素乘积。为了保证目标位置仍为单位增益，采用

$$
\mathbf w_{\mathrm{Cheb}}
=
\frac{\widetilde{\mathbf s}_0}
{\mathbf s_0^H\widetilde{\mathbf s}_0}.
$$

对应代码为

```matlab
spaceWindow = chebwin(N, 30);
timeWindow = chebwin(M, 30);
spaceTimeWindow = kron(spaceWindow, timeWindow);

wChebRaw = sTarget .* spaceTimeWindow;
wCheb = wChebRaw / (sTarget' * wChebRaw);
```

第二幅图中的左图使用 $$\mathbf w_{\mathrm{uniform}}$$，右图使用 $$\mathbf w_{\mathrm{Cheb}}$$。Chebyshev 加窗的主要作用是压低空间维和慢时间维的旁瓣，但通常会以主瓣展宽为代价。

这里需要与真正的 STAP 区分：Chebyshev 加窗的权重由预先给定的窗函数决定，不使用干扰协方差矩阵 $$\mathbf R$$；STAP 权重则由 $$\mathbf R^{-1}\mathbf s_0$$ 自适应决定。因此，第二幅图是**固定空时加窗的二维方向图比较**，第一幅图才是**基于干扰协方差的 STAP 自适应处理结果**。

---

## 十、常规匹配权重与 STAP 权重的区别

如果完全不利用干扰协方差，只对目标空时导向矢量进行匹配，可采用

$$
\boxed{
\mathbf w_{\mathrm{MF}}
=
\frac{
\mathbf s_0
}{
\mathbf s_0^H\mathbf s_0
}.
}
$$

它保证

$$
\mathbf w_{\mathrm{MF}}^H\mathbf s_0=1,
$$

但没有根据杂波和干扰机的统计结构进行自适应调整。

STAP 则使用

$$
\boxed{
\mathbf w_{\mathrm{STAP}}
=
\frac{
\mathbf R^{-1}\mathbf s_0
}{
\mathbf s_0^H
\mathbf R^{-1}\mathbf s_0
}.
}
$$

二者的区别可以概括为：

$$
\text{匹配权重：只知道目标在哪里}
$$

而

$$
\text{STAP：既知道目标在哪里，又利用 } \mathbf R
\text{ 描述干扰主要分布在哪里}.
$$

因此在强杂波和强干扰环境下，STAP 通常能够获得显著更高的输出 SINR。

---

## 十一、实际 STAP 中的协方差估计

本文程序直接按照理论模型构造

$$
\mathbf R_c,\quad
\mathbf R_j,\quad
\mathbf R_n,
$$

因此属于“已知理论协方差”的原理演示。

实际雷达中一般并不知道真实的

$$
\mathbf R.
$$

通常利用待检测距离单元附近、不含目标的 $L$ 个训练距离单元

$$
\mathbf x_1,\ldots,\mathbf x_L
$$

构造样本协方差矩阵

$$
\boxed{
\widehat{\mathbf R}
=
\frac{1}{L}
\sum_{\ell=1}^{L}
\mathbf x_\ell\mathbf x_\ell^H.
}
$$

然后采用

$$
\boxed{
\widehat{\mathbf w}
=
\frac{
\widehat{\mathbf R}^{-1}\mathbf s_0
}{
\mathbf s_0^H
\widehat{\mathbf R}^{-1}
\mathbf s_0
}.
}
$$

当训练样本不足、协方差病态或环境非均匀时，通常还会采用对角加载、降维 STAP、稀疏恢复、结构化协方差估计等方法。

因此，从理论到实际 STAP 的核心链条是

$$
\boxed{
\text{空时数据}
\rightarrow
\widehat{\mathbf R}
\rightarrow
\widehat{\mathbf R}^{-1}\mathbf s_0
\rightarrow
\text{空时自适应抑制}
}
$$

而本文代码首先使用理论 $\mathbf R$，目的是把 STAP 最基本的数学关系完整、清楚地展示出来。

---

## 十二、代码与理论的对应关系

| 理论量 | 数学表达 | MATLAB 变量 |
|---|---|---|
| 空间导向矢量 | $\mathbf a(\theta)$ | `aTarget`, `aClutter`, `aJammer` |
| 多普勒导向矢量 | $\mathbf b(\nu)$ | `bTarget`, `bClutter` |
| 目标空时导向矢量 | $\mathbf a\otimes\mathbf b$ | `sTarget` |
| 杂波导向矩阵 | $\mathbf V_c$ | `Vclutter` |
| 杂波协方差 | $\mathbf R_c$ | `Rc` |
| 干扰机协方差 | $\mathbf R_j$ | `Rj` |
| 白噪声协方差 | $\sigma_n^2\mathbf I$ | `Rn` |
| 总干扰加噪声协方差 | $\mathbf R_n+\mathbf R_c+\mathbf R_j$ | `Rcase{4}` |
| STAP 权重 | $\mathbf R^{-1}\mathbf s_0/(\mathbf s_0^H\mathbf R^{-1}\mathbf s_0)$ | `Wstap{iCase}` |
| 二维空时响应 | $|\mathbf w^H\mathbf s(\theta,\nu)|$ | `responseDB{iCase}` |

```matlab
%% STAP_demo.m
% 空时自适应处理（STAP）标准理论仿真
%
% 本程序采用如下空时数据堆叠方式：
%   x = [x_1^T, x_2^T, ..., x_N^T]^T
% 其中 x_n 为第 n 个阵元在 M 个脉冲上的慢时间数据。
% 因此空时导向矢量统一写为：
%   s(theta,nu) = a(theta) \otimes b(nu)
%
% 本程序重点演示：
%   1) 空间导向矢量与慢时间多普勒导向矢量；
%   2) 旁视阵列条件下的杂波脊；
%   3) 杂波、宽带压制式干扰机和白噪声的协方差建模；
%   4) 最大输出 SINR / MVDR 形式的 STAP 最优权重；
%   5) 图1：四种干扰场景下的 STAP 角度-多普勒二维响应；
%   6) 图2：均匀空时匹配与 Chebyshev 空时加窗的二维响应对比。
%
% 说明：
%   这里直接使用“理论协方差矩阵”构造 STAP 权重，用于清楚展示
%   STAP 的基本原理。实际系统中通常需由邻近训练距离单元估计协方差矩阵。

clear;
clc;
close all;

%% 1. 基本参数

N = 16;                         % 阵元数
M = 16;                         % 相干处理脉冲数
NM = N * M;                     % 空时维数

lambda = 0.03;                  % 工作波长 / m
d = lambda / 2;                 % 阵元间距 / m
v = 60;                         % 平台速度 / (m/s)
Tr = 1e-4;                      % 脉冲重复周期 PRI / s

% 归一化杂波脊斜率：
% nu_c = beta * u
% 其中 u = (d/lambda) sin(theta)，nu = f_d Tr
beta = 2 * v * Tr / d;

% 目标参数
thetaTargetDeg = 10;            % 目标角度 / deg
nuTarget = 0.32;                % 目标归一化多普勒频率 / cycles per PRI

% 噪声、杂波和干扰功率参数
noisePower = 1;                 % 每个空时通道的白噪声功率
CNRdB = 40;                     % 总杂波噪声比 / dB
JNRdB = 35;                     % 每个干扰机的干扰噪声比 / dB

% 杂波离散角度
numClutterPatches = 181;
clutterAngleDeg = linspace(-90, 90, numClutterPatches);

% 两个宽带压制式干扰机，仅具有空间方向性，慢时间近似白
jammerAngleDeg = [-35, 40];

% 二维响应扫描网格
angleScanDeg = -90:1:90;
nuScan = -0.5:0.005:0.5;

%% 2. 目标空时导向矢量

aTarget = spatialSteeringVector(N, d, lambda, thetaTargetDeg);
bTarget = dopplerSteeringVector(M, nuTarget);

% 按照当前堆叠方式，空时导向矢量为 a \otimes b
sTarget = kron(aTarget, bTarget);

%% 3. 白噪声协方差矩阵

Rn = noisePower * eye(NM);

%% 4. 杂波协方差矩阵

% 旁视阵列、静止地面杂波条件下：
%   u(theta) = (d/lambda) sin(theta)
%   nu_c(theta) = beta * u(theta)
clutterSpatialFreq = (d / lambda) * sind(clutterAngleDeg);
clutterDoppler = beta * clutterSpatialFreq;

% 将总杂波功率均匀分配到所有独立杂波块
totalClutterPower = noisePower * 10^(CNRdB / 10);
clutterPatchPower = totalClutterPower / numClutterPatches;

Vclutter = zeros(NM, numClutterPatches);

for k = 1:numClutterPatches
    aClutter = spatialSteeringVector( N, d, lambda, clutterAngleDeg(k));
    bClutter = dopplerSteeringVector( M, clutterDoppler(k));
    Vclutter(:, k) = kron(aClutter, bClutter);
end

% 各杂波块相互不相关且等功率：
%   Rc = sum_k sigma_c,k^2 s_c,k s_c,k^H
%      = sigma_c^2 Vc Vc^H
Rc = clutterPatchPower * (Vclutter * Vclutter');

% 数值上强制 Hermitian，对理论结果无影响
Rc = (Rc + Rc') / 2;

%% 5. 宽带压制式干扰机协方差矩阵

% 对于第 q 个宽带压制式干扰机：
%   Rj,q = Pj (a_j a_j^H) \otimes I_M
%
% 该模型表示：
%   - 空间上来自固定到达角；
%   - 慢时间上近似白，因此会占据该角度处的一整条多普勒方向。
Rj = zeros(NM);
jammerPower = noisePower * 10^(JNRdB / 10);
for q = 1:numel(jammerAngleDeg)
    aJammer = spatialSteeringVector(N, d, lambda, jammerAngleDeg(q));
    Rj = Rj + jammerPower * kron(aJammer * aJammer', eye(M));
end

Rj = (Rj + Rj') / 2;

%% 6. 四种场景下的干扰加噪声协方差矩阵
Rcase = cell(4, 1);
caseName = cell(4, 1);

Rcase{1} = Rn;
caseName{1} = '仅白噪声';

Rcase{2} = Rn + Rc;
caseName{2} = '白噪声 + 杂波';

Rcase{3} = Rn + Rj;
caseName{3} = '白噪声 + 干扰机';

Rcase{4} = Rn + Rc + Rj;
caseName{4} = '白噪声 + 杂波 + 干扰机';

%% 7. 计算 STAP 最优权重与二维角度-多普勒响应
Wstap = cell(4, 1);
responseDB = cell(4, 1);

for iCase = 1:4

    R = Rcase{iCase};

    % 最大输出 SINR 的权重方向为：
    %   w \propto R^{-1} s0
    %
    % 加入无失真约束 w^H s0 = 1 后，得到 MVDR/STAP 权重：
    %   w = R^{-1}s0 / (s0^H R^{-1}s0)
    %
    % 使用反斜杠求解线性方程，避免显式计算 inv(R)。
    x = R \ sTarget;
    Wstap{iCase} = x / (sTarget' * x);

    response = computeSpaceTimeResponse(Wstap{iCase}, N, M, d, lambda, angleScanDeg, nuScan);

    % 因为采用 w^H s0 = 1 的归一化，理论目标响应为 0 dB。
    responseDB{iCase} = 20 * log10(max(response, eps));
    responseDB{iCase}(responseDB{iCase} < -80) = -80;
end

%% 8. 绘制四种场景的角度-多普勒响应
figure('Color', 'w');
for iCase = 1:4
    subplot(2, 2, iCase);
    imagesc(angleScanDeg, nuScan, responseDB{iCase});
    axis xy;
    axis tight;

    xlabel('角度 \theta / deg');
    ylabel('归一化多普勒 \nu = f_d T_r');
    title(caseName{iCase});
    colorbar;
    caxis([-80, 0]);

    hold on;

    % 标出目标位置
    plot(thetaTargetDeg, nuTarget, 'wo', 'MarkerSize', 7, 'LineWidth', 1.5);

    % 对含杂波的场景叠加理论杂波脊
    if iCase == 2 || iCase == 4
        nuRidge = beta * (d / lambda) * sind(angleScanDeg);
        validIndex = abs(nuRidge) <= 0.5;
        plot(angleScanDeg(validIndex), nuRidge(validIndex), 'w--', 'LineWidth', 1.0);
    end
    % 对含干扰机的场景标出干扰机到达角
    if iCase == 3 || iCase == 4
        for q = 1:numel(jammerAngleDeg)
            xline(jammerAngleDeg(q), 'w:', 'LineWidth', 1.0);
        end
    end

    hold off;
end

sgtitle('图1  STAP 在不同干扰环境下的角度-多普勒二维响应');

%% 9. 图2：均匀空时匹配与 Chebyshev 空时加窗对比

% 这一部分不是 STAP 的协方差自适应权重设计，而是固定空时加窗。
% 其目的与经典示例中的第二幅图一致：比较不加窗与 Chebyshev 加窗
% 后的角度-多普勒二维旁瓣特性。
%
% 未加窗空时匹配权重：
%   w_uniform = s0 / (s0^H s0)
%
% 空时 Chebyshev 窗：
%   g = g_space \otimes g_time
%
% 加窗后：
%   w_cheb ∝ s0 .* g
%
% 两种权重都归一化到目标方向单位增益，便于公平比较。

% 未加窗的空时匹配权重
wUniform = sTarget / (sTarget' * sTarget);

% 分别在空间维和慢时间维构造 Chebyshev 窗
spaceWindow = chebwin(N, 30);
timeWindow = chebwin(M, 30);

% 联合空时窗，与 s = a \otimes b 的堆叠方式保持一致
spaceTimeWindow = kron(spaceWindow, timeWindow);

% 将固定空时窗作用在目标空时导向矢量上
wChebRaw = sTarget .* spaceTimeWindow;

% 归一化，使目标位置满足 w^H s0 = 1
wCheb = wChebRaw / (sTarget' * wChebRaw);

% 计算未加窗和加窗后的二维响应
responseUniform = computeSpaceTimeResponse(wUniform, N, M, d, lambda, angleScanDeg, nuScan);

responseCheb = computeSpaceTimeResponse(wCheb, N, M, d, lambda, angleScanDeg, nuScan);

responseUniformDB = 20 * log10(max(responseUniform, eps));
responseChebDB = 20 * log10(max(responseCheb, eps));

responseUniformDB(responseUniformDB < -80) = -80;
responseChebDB(responseChebDB < -80) = -80;

figure('Color', 'w');

subplot(1, 2, 1);
imagesc(angleScanDeg, nuScan, responseUniformDB);
axis xy;
axis tight;
xlabel('角度 \theta / deg');
ylabel('归一化多普勒 \nu = f_d T_r');
title('均匀空时匹配');
colorbar;
caxis([-80, 0]);

subplot(1, 2, 2);
imagesc(angleScanDeg, nuScan, responseChebDB);
axis xy;
axis tight;
xlabel('角度 \theta / deg');
ylabel('归一化多普勒 \nu = f_d T_r');
title('Chebyshev 空时加窗');
colorbar;
caxis([-80, 0]);

sgtitle('图2  空时 Chebyshev 加窗前后的二维响应对比');

%% 10. 输出 SINR 对比

% 组合场景：白噪声 + 杂波 + 干扰机
Rin = Rn + Rc + Rj;

% 常规空时匹配权重，只对准目标，不利用干扰协方差
wMatched = sTarget / (sTarget' * sTarget);

% STAP 最优权重
wSTAP = Wstap{4};

% 设目标功率为 1，则输出 SINR 为：
%   SINR_out = |w^H s0|^2 / (w^H Rin w)
sinrMatched = abs(wMatched' * sTarget)^2 / real(wMatched' * Rin * wMatched);

sinrSTAP = abs(wSTAP' * sTarget)^2 / real(wSTAP' * Rin * wSTAP);

fprintf('--------------------------------------------------\n');
fprintf('组合场景输出 SINR 对比\n');
fprintf('常规空时匹配权重： %.2f dB\n', 10 * log10(sinrMatched));
fprintf('STAP 最优权重：     %.2f dB\n', 10 * log10(sinrSTAP));
fprintf('STAP SINR 增益：    %.2f dB\n', ...
    10 * log10(sinrSTAP / sinrMatched));
fprintf('--------------------------------------------------\n');

%% 局部函数

function a = spatialSteeringVector(N, d, lambda, thetaDeg)
%SPATIALSTEERINGVECTOR 生成 ULA 空间导向矢量
%   a(theta) =
%   [1, exp(j2piu), ..., exp(j2pi(N-1)u)]^T
%   其中：
%       u = (d/lambda) sin(theta)

    elementIndex = (0:N-1).';
    u = (d / lambda) * sind(thetaDeg);

    a = exp(1j * 2 * pi * elementIndex * u);
end


function b = dopplerSteeringVector(M, nu)
%DOPPLERSTEERINGVECTOR 生成慢时间多普勒导向矢量
%
%   b(nu) =
%   [1, exp(j2pinu), ..., exp(j2pi(M-1)nu)]^T
%
%   nu = f_d Tr 为归一化多普勒频率，单位为 cycles per PRI。

    pulseIndex = (0:M-1).';

    b = exp(1j * 2 * pi * pulseIndex * nu);
end


function response = computeSpaceTimeResponse(w, N, M, d, lambda, angleScanDeg, nuScan)
%COMPUTESPACETIMERESPONSE 计算角度-多普勒二维空时响应
%
%   response(iNu,iTheta)
%       = |w^H s(theta_i,nu_j)|
%
%   其中：
%       s(theta,nu) = a(theta) \otimes b(nu)
    numAngle = numel(angleScanDeg);
    numDoppler = numel(nuScan);

    response = zeros(numDoppler, numAngle);
    % 预先构造所有慢时间多普勒导向矢量
    pulseIndex = (0:M-1).';
    Bscan = exp(1j * 2 * pi * pulseIndex * nuScan);

    for iAngle = 1:numAngle
        aScan = spatialSteeringVector(N, d, lambda, angleScanDeg(iAngle));
        % Sscan 的每一列分别为：
        %   a(theta_i) \otimes b(nu_j)
        Sscan = kron(aScan, Bscan);
        response(:, iAngle) = abs(w' * Sscan).';
    end
end

```

