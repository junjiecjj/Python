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


%% 实例：近似点算法解 LASSO 问题
% 对于 LASSO 问题 
%
% $$\min_x \mu\|x\|_1+\frac{1}{2}\|Ax-b\|_2^2,$$
%
% 其对应于等价问题
%
% $$ \min_{x,y} f(x,y)=\mu\|x\|_1+\frac{1}{2}\|y\|_2^2,\quad \mathrm{s.t.}\quad Ax-y-b=0.$$
%
% 利用近似点算法进行求解。对于近似点子问题利用梯度法不精确求解，并考虑半光滑牛顿法加速。
%% 构造 LASSO 问题
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
% 生成随机的矩阵 $A$ 和向量 $u$ 以使得 $b=Au$。正则化系数 $\mu=10^{-3}$。随机迭代初始点。
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
x0 = randn(n, 1);
mu = 1e-3;
%% 利用近似点算法（PPA）求解 LASSO 问题
%%%
% 首先在更严格的停机准则下进行试验，将收敛时得到的历史最优函数值作为真实的最优值的参考 $f^*$。
opts = struct();
opts.verbose = 0;
opts.maxit = 20;
opts.ftol = 1e-14;
opts.gtol = 1e-10;
[x, out] = LASSO_ppa(x0, A, b, mu, opts);
f_star = min(out.fvec);
%%%
% 利用 PPA 求解 LASSO 问题。
opts = struct();
opts.verbose = 0;
opts.ftol = 2e-8;
opts.gtol = 2e-6;
opts.maxit = 10;
opts.step_type = 'fixed';
[x, out] = LASSO_ppa(x0, A, b, mu, opts);
data1 = (out.fvec - f_star)/f_star;
k1 = length(data1);

%%%
% 令正则化系数 $\mu=10^{-2}$，重复实验。在更严格的停机准则下进行试验，
% 将收敛时得到的历史最优函数值作为真实的最优值的参考 $f^*$。
mu = 1e-2;
opts = struct();
opts.verbose = 0;
opts.maxit = 20;
opts.ftol = 1e-14;
opts.gtol = 1e-10;
[x, out] = LASSO_ppa(x0, A, b, mu, opts);
f_star = min(out.fvec);
%%%
% 利用 PPA 求解 LASSO 问题。
opts = struct();
opts.verbose = 0;
opts.ftol = 2e-8;
opts.gtol = 2e-6;
opts.maxit = 10;
mu = 1e-2;
opts.step_type = 'fixed';
[x, out] = LASSO_ppa(x0, A, b, mu, opts);
data2 = (out.fvec - f_star)/f_star;
k2 = length(data2);

%% 结果可视化
% 对每一步的函数值与最优函数值之间的相对误差进行可视化。
fig = figure;
semilogy(0:k1-1, data1, '-o', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:k2-1, data2, '-.*','Color',[0.99 0.1 0.2], 'LineWidth',2);
set(gca,'xtick',1:max(k1,k2));
legend('\mu = 10^{-3}','\mu = 10^{-2}');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','ppa.eps');
%% 结果分析
% 上图展示了在 LASSO 问题上近似点法的数值表现。固定步长时，每一次外层迭代，
% 目标函数值都有明显下降，外层迭代仅 $4$ 次目标函数已经接近收敛。
%
% 注意，每一次外层迭代都需要求解关于 $z$ 的子问题，算法的主要运算量在内迭代子问题的求解上。
% 因此有必要对子问题使用半光滑牛顿法进行加速。
%% 参考页面
% 算法请参考 <LASSO_ppa.html LASSO问题的近似点算法>。
%
% 此页面的源代码请见： <../download_code/lasso_ppa/demo_ppa.m demo_ppa.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将