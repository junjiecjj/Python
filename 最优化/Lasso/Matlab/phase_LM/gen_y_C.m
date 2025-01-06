%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Copyright (C) 2020 Zaiwen Wen, Haoyang Liu, Jiang Hu
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Change log:
%
%   2017.6.10 (Chao Ma):
%     First version
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 信号采集函数
% 对于输入的信号 $x$利用编码衍射模型，得到其信号采集
%
% $$ \displaystyle b_j=\left| \sum_{t=0}^{n-1}x_t\bar{b}_\ell(t)e^{-i2\pi kt/n}
% \right|^2,\quad j=(l,k),0\le k\le n-1,1\le\ell\le L,$$
%
% 其中 $b_j$ 表示采集的波形 $d_\ell$下信号 $\{x_t\}$的衍射图的模长。通过改变 $\ell$和对应的波形
% $d_\ell$我们生成一系列编码衍射图。
%% 函数主体
% 函数要求给定信号 $x$ 和信号采集次数 $L$, 返回记录波形的矩阵
% $A\in\mathcal{C}^{n,L}=(d_1,d_2,\dots,d_L)$ 和采集到的信号
% $y\in\mathcal{C}^{nL}$。
function [ y, A ] = gen_y_C( x, L )
n=size(x,1);
A=zeros(n,L);
y=zeros(n*L,1);
%%%
% $L$ 为信号采集的次数。 $c_1$ 分别依概率 $\frac{1}{4}$ 取 $+1$, $-1$, $+i$, $-i$。
for k=1:L
    b1=zeros(n,1);b2=b1;
    b=rand(n,1);
    b1(b<0.25)=1;
    b1(b>=0.25 & b<0.5)=-1;
    b1(b>=0.5 & b<0.75)=-1i;
    b1(b>=0.75)=1i;
    %%%
    % $c_2$ 依概率 $\frac{4}{5}$ 取 $\frac{\sqrt{2}}{2}$，依概率 $\frac{1}{5}$ 取
    % $\sqrt{3}$。
    b=rand(n,1);
    b2(b<0.8)=sqrt(2)/2;
    b2(b>=0.8)=sqrt(3);
    %%%
    % 以独立同分布的随机向量 $d_\ell,\ell=0,1,\dots,L$模拟实际场景中的波形，
    % 其中 $d_\ell(t)=c_1c_2$。
    %
    % 信号采集，利用快速傅里叶变换
    % <https://www.mathworks.com/help/matlab/ref/fft.html fft>
    % 函数计算以波形 $d_\ell$
    % 完成上文给定的信号采集，并记录在 $y_{((k-1)n+1,kn)}$中。
    A(:,k)=b1.*b2;
    y((k-1)*n+1:k*n)=abs(fft(x.*conj(A(:,k)))).^2;
end
end
%% 参考页面
% 该函数实现编码衍射模型的信号采集过程，我们在 <demo.html 实例：编码衍射模型> 
% 中展示该模型的一个实例。
%
% 此页面的源代码请见： <../download_code/phase_LM/gen_y_C.m gen_y_C.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将