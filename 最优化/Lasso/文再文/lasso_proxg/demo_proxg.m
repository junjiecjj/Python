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


%% 实例：近似点梯度法和 Nesterov 加速算法求解 LASSO 问题
% 考虑 LASSO 问题
%
% $$ \displaystyle \min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1, $$
%
% 在连续化策略下，分别利用近似点梯度法和两种 Nesterov 加速算法对其求解。
%% 构建 LASSO 优化问题
% 设定随机种子。
clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%%
% 构造 LASSO 优化问题
%
% $$\min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1,$$
%
% 生成随机的矩阵 $A$ 和向量 $u$ 以使得 $b=Au$。给定正则化系数为 |1e-3|。
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
x0 = randn(n, 1);
mu = 1e-3;
%% 求解 LASSO 优化问题
% BB 步长带线搜索的连续化近似点梯度法，首先以更严格的停机准则，得到最优值 $f^*$ 的参考。
AA = A' * A; L = eigs(AA, 1);
opts = struct();
opts.method = 'proximal_grad';
opts.verbose = 0;
opts.maxit = 4000;
opts.opts1 = struct();
opts.opts1.ls = 1;
opts.opts1.bb = 1;
opts.alpha0 = 1/L;
opts.ftol = 1e-12;
opts.gtol = 1e-10;
addpath('../LASSO_con')
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = min(out.fvec);

%%%
% BB 步长带线搜索的连续化近似点梯度法。
opts = struct();
opts.method = 'proximal_grad';
opts.opts1 = struct();
opts.verbose = 0;
opts.maxit = 400;
opts.opts1.ls = 1;
opts.opts1.bb = 1;
opts.alpha0 = 1/L;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data1 = (out.fvec - f_star)/f_star;
k1 = min(length(data1),400);
data1 = data1(1:k1);

%%%
% 固定步长且无线搜索的连续化近似点梯度法。
opts = struct();
opts.method = 'proximal_grad';
opts.opts1 = struct();
opts.verbose = 0;
opts.maxit = 400;
opts.opts1.ls = 0;
opts.opts1.bb = 0;
opts.alpha0 = 1/L;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data2 = (out.fvec - f_star)/f_star;
k2 = min(length(data2),400);
data2 = data2(1:k2);

%%%
% BB 步长带线搜索的 FISTA 算法。
opts = struct();
opts.method = 'Nesterov';
opts.opts1 = struct();
opts.verbose = 0;
opts.maxit = 400;
opts.opts1.ls = 1;
opts.opts1.bb = 1;
opts.alpha0 = 1/L;
opts.ftol0 = 1;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data3 = (out.fvec - f_star)/f_star;
k3 = min(length(data3),400);
data3 = data3(1:k3);

%%%
% 固定步长且无线搜索的 FISTA 算法。
opts = struct();
opts.method = 'Nesterov';
opts.opts1 = struct();
opts.verbose = 0;
opts.maxit = 400;
opts.opts1.ls = 0;
opts.opts1.bb = 0;
opts.alpha0 = 1/L;
opts.ftol0 = 1;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data4 = (out.fvec - f_star)/f_star;
k4 = min(length(data4),400);
data4 = data4(1:k4);

%%%
% 固定步长且无线搜索的第二类 Nesterov 加速算法。
opts = struct();
opts.method = 'Nesterov2nd';
opts.opts1 = struct();
opts.verbose = 0;
opts.maxit = 400;
opts.opts1.ls = 0;
opts.opts1.bb = 0;
opts.alpha0 = 1/L;
opts.ftol0 = 1;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data5 = (out.fvec - f_star)/f_star;
k5 = min(length(data5),400);
data5 = data5(1:k5);

%% 结果可视化
% 在第一张图中，我们展示连续化近似点梯度法在 BB 步长和固定步长下的收敛性的差别。
fig = figure;
semilogy(0:k1-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:k2-1, data2, '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
legend('BB步长', '固定步长');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','proxg.eps');

%%%
% 在第二张图中，我们对比基础的近似点梯度法和其各版本的加速算法的收敛性差别。
fig = figure;
semilogy(0:k1-1, data1, '-', 'Color',[0.99 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:k3-1, data3, '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.2);
hold on
semilogy(0:k4-1, data4, '--','Color',[0.2 0.1 0.99], 'LineWidth',1.5);
hold on
semilogy(0:k5-1, data5, ':','Color',[0.5 0.2 1], 'LineWidth',1.8);
legend('BB步长的近似点梯度法', 'BB步长的FISTA算法','固定步长的FISTA算法', '固定步长的第二类Nesterov算法');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','fproxg.eps');
%% 结果分析
% 左图展示了固定步长和 BB 步长的连续化近似点梯度法的收敛速度，结合了线搜索的 BB
% 步长可以显著提高算法的收敛速度。
%
% 在右图中，我们将近似点梯度法和几种加速算法进行比较。在固定步长下，FISTA
% 算法相较于第二类 Nesterov 算法稍快，也注意到这里的 FISTA 算法是非单调算法。
% BB 步长有效地加快收敛，此外，
% 带有线索搜地近似点梯度法可以比同样设定的 FISTA 算法更快收敛。
%% 参考页面
% 在此页面中的算法请参考：
%
% * <../LASSO_con/LASSO_con.html
% LASSO 问题的连续化策略>
% * <LASSO_proximal_grad_inn.html
% LASSO 问题的近似点梯度法>
% * <LASSO_Nesterov_inn.html LASSO 问题的 FISTA
% 算法>
% * <LASSO_Nesterov2nd_inn.html LASSO 问题的第二类 Nesterov 加速算法>
%
% 此页面的源代码请见： <../download_code/lasso_proxg/demo_proxg.m demo_proxg.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将