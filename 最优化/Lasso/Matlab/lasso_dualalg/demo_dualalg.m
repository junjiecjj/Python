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


%% 实例：利用原始-对偶混合梯度 (PDHG) 算法解 LASSO 问题
% 利用 PDHG 方法优化 LASSO 问题
%
% $$ \displaystyle\min_x \frac{1}{2}\|Ax-b\|_2^2 + \mu \|x\|_1,$$
%
% 并采用连续化策略，采用更大的正则化系数并随着迭代逐渐衰减至原值。
%% 构造LASSO问题
% 设定随机种子。
clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%%
% 构造 LASSO 优化问题
%
% $$\displaystyle \min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1.$$
%
% 生成随机的矩阵 $A$ 和向量 $u$ 以使得 $b=Au$。正则化系数 $\mu=1e-3$。随机迭代初始点。
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
x0 = randn(n, 1);
mu = 1e-3;
%% 利用原始-对偶混合梯度 (PDHG) 算法求解 LASSO 问题
% 首先在更严格的停机准则下进行试验，将收敛时得到的历史最优函数值作为真实的最优值的参考 $f^*$。
AA = A' * A; L = eigs(AA, 1);
opts = struct();
opts.method = 'pdhg';
opts.verbose = 0;
opts.maxit = 5000;
opts.ftol = 1e-10;
opts.gtol = 1e-8;
opts.fol0 = 1e6;
opts.gtol0 = 1e6;
opts.alpha0 = 1/L;
opts.opts1.cp = 1;
addpath('../LASSO_con')
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = min(out.fvec);

%%%
% 利用 PDHG 求解 LASSO 问题。
opts.verbose = 0;
opts.ftol = 1e-8;
opts.gtol = 1e-6;
opts.maxit = 800;
opts.alpha0 = 1/L;
opts.opts1.cp = 0;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data1 = (out.fvec - f_star)/f_star;
k1 = length(data1);

%%%
% 利用 Chambolle-Pock 方法求解 LASSO 问题，利用 |opts.opts1.cp| 标记是否采用 Chambolle-Pock
% 方法。
opts.verbose = 0;
opts.maxit = 800;
opts.alpha0 = 1/L;
opts.opts1.cp = 1;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data2 = (out.fvec - f_star)/f_star;
k2 = length(data2);

%% 结果可视化
% 对每一步的目标函数值与最优函数值的相对误差进行可视化。
fig = figure;
semilogy(0:k1-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:k2-1, data2, '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
legend('PDHG','Chambolle-Pock');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','dualalg.eps');
%% 结果分析
% 上图展示了 PDHG 和 Chambolle-Pock 算法在 LASSO 问题上的表现。
% 在固定步长和采用连续化策略的情况下，算法达到了收敛。
% 并且注意到尽管 PDHG 在某些情况下没有收敛性保证，在这个例子上 PDHG 的速度比 Chambolle-Pock 略快。
%% 参考页面
% 关于 PDHG 和 Chambolle-Pock 算法请参考页面
% <LASSO_pdhg_inn.html LASSO问题的原始-对偶混合梯度算法>，LASSO 问题的连续化策略参考
% <..\LASSO_con\LASSO_con.html LASSO 问题连续化策略>。
%
% 此页面的源代码请见： <../download_code/lasso_dualalg/demo_dualalg.m
% demo_dualalg.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将