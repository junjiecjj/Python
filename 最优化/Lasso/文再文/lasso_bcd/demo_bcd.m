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


%% 实例：利用分块坐标下降法（BCD）求解 LASSO 问题
% 对于 LASSO 问题 
%
% $$\min_x \mu\|x\|_1+\frac{1}{2}\|Ax-b\|_2^2,$$
%
% 考虑 $x$ 的每一个坐标分量为一个坐标块，利用分块坐标下降法求解该问题。

%% 构造 LASSO 问题
% 设定随机种子。
clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%%
% 构造 LASSO 优化问题
%
% $$\min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1.$$
%
% 生成随机的矩阵 $A$ 和向量 $u$ 以使得 $b=Au$。正则化系数 $\mu=10^{-3}$。 随机迭代初始点。
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
x0 = randn(n, 1);
mu = 1e-3;

%% 利用 BCD 求解 LASSO 问题
%%%
% 首先在更严格的停机准则下进行试验，将收敛时得到的历史最优函数值作为真实的最优值的参考 $f^*$。
opts = struct();
opts.method = 'bcd';
opts.verbose = 0;
opts.maxit = 100;
opts.ftol = 1e-14;
opts.gtol = 1e-8;
opts.ftol0 = 1e6;
opts.gtol0 = 1e6;
addpath('../LASSO_con')
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = min(out.fvec);

%%%
% 利用 BCD 求解 LASSO 问题。
opts = struct();
opts.method = 'bcd';
opts.verbose = 0;
opts.ftol = 1e-8;
opts.gtol = 1e-6;
opts.ftol0 = 1e6;
opts.gtol0 = 1e6;
opts.maxit = 10;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data1 = (out.fvec - f_star)/f_star;
k1 = length(data1);

%%%
% 将正则化系数改为 $\mu=10^{-2}$ 重复实验。首先在更严格的停机准则下进行试验，
% 将收敛时得到的历史最优函数值作为真实的最优值的参考 $f^*$。
mu = 1e-2;
opts = struct();
opts.method = 'bcd';
opts.verbose = 0;
opts.maxit = 100;
opts.ftol = 1e-14;
opts.gtol = 1e-8;
opts.ftol0 = 1e6;
opts.gtol0 = 1e6;
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = min(out.fvec);

%%%
% 利用 BCD 求解 LASSO 问题。
opts = struct();
opts.method = 'bcd';
opts.verbose = 0;
opts.ftol = 1e-8;
opts.gtol = 1e-6;
opts.ftol0 = 1e6;
opts.gtol0 = 1e6;
opts.maxit = 10;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data2 = (out.fvec - f_star)/f_star;
k2 = length(data2);

%% 结果可视化
% 对当前函数值与最优函数值的相对误差进行可视化。
fig = figure;
semilogy(0:k1-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:k2-1, data2, '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
legend('\mu = 10^{-3}','\mu = 10^{-2}');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','bcd2.eps');
%% 结果分析
% 上图展示了结合连续化策略的 BCD 方法在 LASSO 问题上的表现。BCD 方法很快地收敛到问题的解。
% 相比其它算法，坐标下降法具有不需要调节步长参数的优势。
%% 参考页面
% BCD 算法请参考 <LASSO_bcd_inn.html LASSO 问题的分块坐标下降法解法>。
% LASSO 问题的连续化策略框架参考
% <..\LASSO_con\LASSO_con.html LASSO 问题连续化策略框架>。
%
% 此页面的源代码请见： <../download_code/lasso_bcd/demo_bcd.m demo_bcd.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将