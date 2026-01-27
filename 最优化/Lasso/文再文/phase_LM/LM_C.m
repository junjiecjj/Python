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

%% 编码衍射模型的 LM 算法
% 考虑非线性最小二乘问题：
%
% $$\min_z \frac{1}{2}\|r(z)\|^2_2,$$
%
% 其中 $r$ 为非线性映射。
%
% LM 方法本质上是一种信赖域型的方法。在信赖域方法中，每一步求下面的子问题
%
% $$ \min_d \frac{1}{2}\|J^kd+r^k\|_2^2,\ \mathrm{s.t.} \|d\|\le\Delta_k, $$
%
% 其中 $J^k$ 为 $z^k$ 处的雅可比矩阵。
% LM 方法通过引入正则化将信赖域半径的约束罚到目标函数中，求解如下LM 方程
%
% $$ (\bar{J}(\mathbf{z}^k)^\top J(\mathbf{z}^k)+\lambda_k)d^k=-\nabla f(\mathbf{z}^k). $$
%
% 对于编码衍射模型中，我们有
%
% $$ r_{k,\ell}(z)=\displaystyle\frac{1}{2}\left(
% \left|\sum_{t=0}^{n-1}z_t\bar{d}_\ell(t)e^{-i2\pi kt/n} \right|^2 
% - b_{k,\ell} \right), $$
%
% 其中 $b_j=\displaystyle\left|\sum_{t=0}^{n-1}x_t\bar{d}_\ell (t)e^{-i2\pi
% kt/n}\right|,$ $x$ 为真实信号。
%% 初始化和迭代准备
% 输入信息：迭代初始点 $z_0$，观测模长 $y$，采样信号 $A$，原始数据 $x$ 和停机准则 |stop_cri| 。
% 输出信息： 迭代得到的解 $z$（其为真实信号 $x$的恢复），包含迭代信息的结构体 |info| 。
%
% * |info.eff|：每一步迭代的相对误差
% * |info.time|：每一步迭代的 cpu 时间
% * |info.prod|：矩阵向量积的次数
% * |info.ite|：总迭代次数
% * |info.state|：标记是否收敛
function [ z, info ] = LM_C( z0, y, A, x, acu, stop_cri )
%%%
% 参数设定，最大迭代次数 50(25) 次，并计时。
%
% 利用截断共轭梯度法求解LM方程，设定截断共轭梯度法的最大迭代次数。
T=cputime;
ite_max=50;
if acu==1
    ite_max=25;
end

if acu==1
    CG_ite_max=50;
else
    CG_ite_max=5;
end
%%%
% 初始化，迭代次数置零，以 $z^0$ 为迭代初始点。
ite=0;
z=z0;
%%%
% 将向量化的采集信号 $y$ 转化为 $n\times L$ 的矩阵。
[n,L]=size(A);
m=n*L;
y=reshape(y,n,L);
%%%
% 相对误差 $r$：
%
% $$ r=\displaystyle\min_{\phi\in[0,2\pi]}\|x-ze^{i\phi}\|_F/\|x\|_F=
% \|x-ze^{-\theta i}\|_F/\|x\|_F, $$
%
% 其中 $\theta=\mathrm{angle}\sum_{i=1}^n(\bar{x}_iz_i)$ 为 $x$ 与
% $z$ 之间的夹角。当 $z=e^{i\phi}x$ 时，该相对误差为 $0$。
relerr=norm(x-exp(-1i*angle(trace(x'*z)))*z,'fro')/norm(x,'fro');
%%%
% 初始化输出信息， |info.err| 记录当前迭代步对应的相对误差，
% |info.time| 记录每一迭代步的 cpu 时间， |info.prod| 表示矩阵和向量乘积的次数。
info.err=relerr;
info.time=[0];
info.prod=[0];
prod=0;
%% 迭代主体
% 达到收敛条件：相对误差 $r$ 小于阈值 |stop_cri| ，或者迭代次数超过最大迭代次数 
% |ite_max| 限制，停止迭代。
while relerr>stop_cri && ite<ite_max
    %%%
    % 利用共轭梯度法进行近似求解得到 $d^k$。进行一步迭代 $z^{k+1}=z^k-d^k$。
    [solu,in]=CG_CDP(z,A,y,CG_ite_max,acu);
    z=z-solu(1:n);
    
    %%%
    % 迭代步数加一，更新迭代信息，记录当前步的相对误差、cpu 时间、矩阵向量积次数。
    ite=ite+1;
    relerr=norm(x-exp(-1i*angle(trace(x'*z)))*z,'fro')/norm(x,'fro');
    info.err=[info.err,relerr];
    info.time=[info.time,cputime-T];
    prod=prod+in.prod;
    info.prod=[info.prod,prod];
end
%%%
% 当迭代退出时，记录总迭代步。并以 |info.state| 记录退出原因（是否达到收敛而退出迭代）。
info.ite=ite;
if ite==ite_max
    info.state=0;
else
    info.state=1;
end
end
%% 参考页面
% 我们在页面 <demo.html 实例：编码衍射模型>
% 中展示该算法的一个应用，并将其与其它算法进行比较。算法使用预条件共轭梯度法求解LM方程，
% 参考 <CG_CDP.html 编码衍射模型 LM 子问题的预条件共轭梯度算法>。
%
% 此页面的源代码请见： <../download_code/phase_LM/LM_C.m LM_C.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将