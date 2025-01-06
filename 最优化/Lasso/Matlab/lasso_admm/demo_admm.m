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


%% 实例：交替方向乘子法（ADMM）解 LASSO 问题
% 考虑LASSO 问题 
% 
% $$\min_x \mu\|x\|_1+\frac{1}{2}\|Ax-b\|_2^2.$$
% 
% 首先考虑利用 ADMM 求解原问题：将其转化为 ADMM 标准问题
%
% $$ 
% \begin{array}{rl}
% \displaystyle\min_{x,z} & \hspace{-0.5em}\frac{1}{2}\|Ax-b\|^2_2+\mu\|z\|_1,\\
% \displaystyle\mathrm{s.t.} & \hspace{-0.5em}x=z,
% \end{array} 
% $$
%
% 则可以利用 ADMM 求解。相应的，对于 LASSO 对偶问题
%
% $$ 
% \begin{array}{rl}
% \displaystyle\min_y & \hspace{-0.5em}b^\top y+\frac{1}{2}\|y\|_2^2,\\
% \displaystyle\mathrm{s.t.} & \hspace{-0.5em}\|A^\top y\|_\infty\le\mu,
% \end{array}
% $$
%
% 则等价于
%
% $$ 
% \begin{array}{rl}
% \displaystyle\min_{y,z} & \hspace{-0.5em}b^\top y +\frac{1}{2}\|y\|_2^2+I_{\|z\|_\infty\le\mu}(z),\\
% \displaystyle\mathrm{s.t.} & \hspace{-0.5em}A^\top y + z = 0.
% \end{array}
% $$
%
% 对于上述的两个等价问题利用 ADMM 求解。
%% 构造 LASSO 问题
% 设定随机种子。
clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%%
% 构造 LASSO 优化问题
% 
% $$\displaystyle\min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1.$$
% 
% 生成随机的矩阵 $A$ 和向量 $u$ 以使得 $b=Au$。 正则化系数 $\mu=10^{-3}$。 随机迭代初始点。
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
x0 = randn(n, 1);
mu = 1e-3;
%% 利用 ADMM 求解 LASSO 问题
% 首先在更严格的停机准则下进行试验，将收敛时得到的历史最优函数值作为真实的最优值的参考 $f^*$。
opts = struct();
opts.verbose = 0;
opts.maxit = 2000;
opts.sigma = 1e-2;
opts.ftol = 1e-12; 
opts.gtol = 1e-15;
[x, out] = LASSO_admm_primal(x0, A, b, mu, opts);
f_star = min(out.fvec); 
%%%
% 利用 ADMM 求解 LASSO 对偶问题。
opts = struct();
opts.verbose = 0;
opts.maxit = 2000;
opts.sigma = 1e2;
opts.ftol = 1e-8; 
opts.gtol = 1e-10;
[x, out] = LASSO_admm_dual(x0, A, b, mu, opts);
data1 = (out.fvec - f_star)/f_star;
k1 = length(data1);
%%%
% 利用 ADMM 求解 LASSO 原问题。
opts = struct();
opts.verbose = 0;
opts.maxit = 2000;
opts.sigma = 1e-2;
opts.ftol = 1e-8; 
opts.gtol = 1e-10;
[x, out] = LASSO_admm_primal(x0, A, b, mu, opts);
data2 = (out.fvec - f_star)/f_star;
k2 = length(data2);
%% 结果可视化
% 对每一步的目标函数值与最优函数值的相对误差进行可视化。
fig = figure;
semilogy(0:k1-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:k2-1, data2, '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
legend('ADMM解对偶问题','ADMM解原问题');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','admm.eps');
%% 结果分析
% 上图反映了使用 ADMM 求解 LASSO 原问题和对偶问题的表现。在两个问题上目标函数值都存在波动，
% 表明 ADMM 并非下降算法，然而在没有使用连续化策略的情况下，ADMM 依然达到了收敛，
% 这说明 ADMM 较之其它算法具有一定的优越性。注意，虽然在这一例子中 ADMM
% 求解原问题所需迭代次数较少，但求解对偶问题时每一步迭代时间更短。综合看来在该例子中 ADMM
% 求解对偶问题的性能更高。
%% 参考页面
% 算法请参考： <LASSO_admm_primal.html 利用交替方向乘子法求解 LASSO 原问题> 和
% <LASSO_admm_dual.html 利用交替方向乘子法求解 LASSO 对偶问题>。
%
% 此页面的源代码请见： <../download_code/lasso_admm/demo_admm.m demo_admm.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将