---
layout: post
title: 比较相位差法和Capon算法在1发2收雷达系统下的差别
date: 2024-09-06 20:39:00
excerpt_type: html
categories: [信号处理算法]
---

对于多通道的雷达，可以通过波束成型或者是一些基于子空间的方法实现多目标分类和高分辨率测角。但是当雷达只有两个通道的时候，直觉上就感觉这些DOA算法都会退化到和相位差法一样。

就像一元线性回归的时候，样本数多我们可以使用最小二乘法，但如果只有两个样本，那最小二乘法就会退化成已知两点求一条直线。类推到雷达，比如Capon就是一种基于最小二乘法的波束成形方法，应该也会退化。子空间法也为直接失效，二维的接收信号空间，那只能分出一个维度作为信号子空间，一个维度作为噪声子空间。

从信息论的角度来说，角度信息只反映在不同通道接收信号的的时间延迟上，基于不同的准则（最大似然、最小方差等）处理不同通道直接的时延差可以活得比单单使用两个通道更优的结果。当雷达只有两个通道的时候，那这两个通道的相位差就是我们能获得的关于目标角度的所有信息量了，那相位差法就已经能得到这种系统下的最优解s了。

---

### **1. 问题背景**

- **相位差法**：利用两个接收天线接收到的信号之间的相位差，计算目标的入射角 $\theta$ 。
- **Capon算法（MVDR算法）**：通过最小化输出功率，同时对特定方向上的信号保持无失真响应，得到空间谱，并在谱峰值处估计目标的入射角 $\theta$ 。

---

### **2. 目标**

证明在一发两收的雷达系统中，Capon算法的空间谱在相位差法计算得到的角度 $\theta$ 处达到峰值。

---

### **3. 基本假设和模型**

#### **3.1. 系统设置**

- **天线排列**：两个接收天线，间距为 $d$ 。
- **信号入射角度**：目标以角度 $\theta_0$ 入射。
- **信号模型**：窄带平面波假设。

#### **3.2. 信号表达式**

- **接收信号的相位差**：
  $$
  \Delta\phi_0 = \frac{2\pi d \sin(\theta_0)}{\lambda}
  $$
- **接收信号向量**：
  $$
  \mathbf{x} = s \mathbf{a}(\theta_0) + \mathbf{n}
  $$
  其中：
  -  $s$ 是目标信号（假设为零均值、方差为 $\sigma_s^2$ 的随机变量）。
  -  $\mathbf{a}(\theta_0)$ 是导向向量。
  -  $\mathbf{n}$ 是噪声向量（假设为零均值、方差为 $\sigma_n^2$ 的高斯白噪声）。

#### **3.3. 导向向量**

对于两个天线，导向向量为：
$$
\mathbf{a}(\theta) = \begin{bmatrix} 1 \\ e^{-j\beta} \end{bmatrix}
$$
其中：
$$
\beta = \frac{2\pi d \sin(\theta)}{\lambda}
$$

---

### **4. 相位差法计算入射角**

相位差法直接测量两个接收信号之间的相位差 $\Delta\phi$ ，然后计算入射角 $\theta$ ：
$$
\theta = \arcsin\left( \frac{\Delta\phi \lambda}{2\pi d} \right)
$$

---

### **5. Capon算法推导**

#### **5.1. 协方差矩阵**

接收信号的协方差矩阵为：
$$
\mathbf{R} = E[\mathbf{x}\mathbf{x}^H] = \sigma_s^2 \mathbf{a}(\theta_0)\mathbf{a}^H(\theta_0) + \sigma_n^2 \mathbf{I}
$$
展开后：
$$
\mathbf{R} = \begin{bmatrix}
\sigma_s^2 + \sigma_n^2 & \sigma_s^2 e^{j\beta_0} \\
\sigma_s^2 e^{-j\beta_0} & \sigma_s^2 + \sigma_n^2
\end{bmatrix}
$$
其中 $\beta_0 = \frac{2\pi d \sin(\theta_0)}{\lambda}$ 。

#### **5.2. 计算协方差矩阵的逆矩阵**

首先计算行列式：
$$
\det(\mathbf{R}) = (\sigma_s^2 + \sigma_n^2)^2 - \sigma_s^4 = 2\sigma_s^2\sigma_n^2 + \sigma_n^4
$$

逆矩阵为：
$$
\mathbf{R}^{-1} = \frac{1}{\det(\mathbf{R})} \begin{bmatrix}
\sigma_s^2 + \sigma_n^2 & -\sigma_s^2 e^{j\beta_0} \\
-\sigma_s^2 e^{-j\beta_0} & \sigma_s^2 + \sigma_n^2
\end{bmatrix}
$$

#### **5.3. Capon空间谱计算**

Capon算法的空间谱定义为：
$$
P(\theta) = \frac{1}{\mathbf{a}^H(\theta)\mathbf{R}^{-1}\mathbf{a}(\theta)}
$$

计算分母：
$$
D(\theta) = \mathbf{a}^H(\theta)\mathbf{R}^{-1}\mathbf{a}(\theta)
$$

展开 $\mathbf{a}(\theta)$ 和 $\mathbf{R}^{-1}$ ：
$$
\mathbf{a}^H(\theta) = \begin{bmatrix} 1 & e^{j\beta} \end{bmatrix}
$$
$$
\mathbf{a}(\theta) = \begin{bmatrix} 1 \\ e^{-j\beta} \end{bmatrix}
$$

计算：
$$
D(\theta) = \begin{bmatrix} 1 & e^{j\beta} \end{bmatrix} \mathbf{R}^{-1} \begin{bmatrix} 1 \\ e^{-j\beta} \end{bmatrix}
$$

将 $\mathbf{R}^{-1}$ 代入，计算结果为：
$$
D(\theta) = \frac{2}{\det(\mathbf{R})} \left[ (\sigma_s^2 + \sigma_n^2) - \sigma_s^2 \cos(\beta_0 - \beta) \right]
$$

#### **5.4. 空间谱的峰值分析**

Capon空间谱为：
$$
P(\theta) = \frac{1}{D(\theta)} = \frac{\det(\mathbf{R})}{2\left[ (\sigma_s^2 + \sigma_n^2) - \sigma_s^2 \cos(\beta_0 - \beta) \right]}
$$

注意到 $\det(\mathbf{R})$ 和分母前的常数对于 $\theta$ 是常数，因此空间谱的变化取决于分母中的 $\cos(\beta_0 - \beta)$ 。

当 $\beta = \beta_0$ 时， $\cos(\beta_0 - \beta) = 1$ ，此时分母最小，空间谱 $P(\theta)$ 达到最大值。

由于 $\beta = \frac{2\pi d \sin(\theta)}{\lambda}$ ，因此当 $\theta = \theta_0$ 时，空间谱达到峰值。

---

### **6. 结论**

- **Capon算法的空间谱在 $\theta = \theta_0$ 处达到峰值**，即在真实的入射角度处。
- **相位差法**也是通过测量相位差 $\Delta\phi = \beta_0$ ，然后计算入射角 $\theta$ 。
- **因此，Capon算法的谱峰值对应的角度与相位差法计算得到的角度相同**。

---

### **7. 总结**

通过上述推导，我们证明了一发两收的雷达系统中，Capon算法的空间谱峰值与相位差法计算得到的入射角度一致。这是因为在只有两个接收天线的情况下，Capon算法的性能等价于利用相位差信息进行测角。

**实质上，两种方法都利用了接收信号之间的相位差信息，只是Capon算法通过波束形成的方式进行了更为复杂的处理。在只有两个天线的情况下，这种复杂处理并未带来额外的测角优势，因此两种方法的结果一致。**

---

### **8. 参考公式**

- **相位差与入射角的关系**：
  $$
  \Delta\phi = \frac{2\pi d \sin(\theta)}{\lambda}
  $$
- **Capon空间谱**：
  $$
  P(\theta) = \frac{1}{\mathbf{a}^H(\theta)\mathbf{R}^{-1}\mathbf{a}(\theta)}
  $$
- **导向向量**：
  $$
  \mathbf{a}(\theta) = \begin{bmatrix} 1 \\ e^{-j\beta} \end{bmatrix}
  $$
- **协方差矩阵**：
  $$
  \mathbf{R} = \sigma_s^2 \mathbf{a}(\theta_0)\mathbf{a}^H(\theta_0) + \sigma_n^2 \mathbf{I}
  $$

---
