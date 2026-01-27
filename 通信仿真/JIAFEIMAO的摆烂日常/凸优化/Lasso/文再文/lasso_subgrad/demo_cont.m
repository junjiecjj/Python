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


%% 实例：连续化次梯度法解 LASSO 问题
% 我们将在此页面中构造一个 LASSO 问题
%
% $$ \displaystyle \min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1. $$
%
% 并且展示连续化次梯度方法在其中的应用。
%% 构造LASSO优化问题
% 设定随机种子。
clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%%
% 构造 LASSO 优化问题
% 
% $$ \displaystyle\min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1. $$
% 
% 
% 生成随机的矩阵 $A$ 和向量 $u$ 以使得 $b=Au$。第一次实验，给定正则化系数为 |1e-3| 。
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
x0 = randn(n, 1);
mu = 1e-3;

%% 求解 LASSO 优化问题
% 固定步长为 $\displaystyle \frac{1}{\lambda_{max}(A^\top A)}$。
AA = A' * A;
L = eigs(AA, 1);
%%%
% 首先在更严格的停机准则下进行试验，将收敛时得到的函数值作为真实的最优值的参考 $f^*$。
opts = struct();
opts.maxit = 5000;
opts.maxit_inn = 500;
opts.opts1 = struct();
opts.method = 'subgrad';
opts.opts1.step_type = 'diminishing';
opts.verbose = 0;
opts.alpha0 = 1/L;
opts.ftol = 1e-12;
opts.ftol0 = 1e4;
opts.etag = 1;
addpath('../LASSO_con')
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = out.fvec(end);

%%%
% 求解 LASSO 问题，记录输出。
opts.maxit = 3000;
opts.maxit_inn = 200;
opts.opts1.step_type = 'diminishing';
opts.verbose = 0;
opts.ftol = 1e-8;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data1 = (out.fvec - f_star) / f_star;
k1 = length(data1);


%%%
% 将 $\mu$ 修改为 |1e-2| 重复实验。
mu = 1e-2;
opts.maxit = 5000;
opts.maxit_inn = 500;
opts.opts1.step_type = 'fixed';
opts.ftol = 1e-10;
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = out.fvec(end);

opts.maxit = 3000;
opts.maxit_inn = 200;
opts.ftol = 1e-8;
opts.opts1.step_type = 'fixed';
[x, out] = LASSO_con(x0, A, b, mu, opts);
data2 = (out.fvec - f_star) / f_star;
k2 = length(data2);
%% 结果可视化
% 可视化优化过程：观察目标函数值随迭代次数的变化。
fig = figure;
semilogy(1:k1, max(data1,0), '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(1:k2, max(data2,0), '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
legend('\mu = 10^{-3}', '\mu = 10^{-2}');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','subgrad.eps');
%% 结果分析
% 于固定正则化系数 $\mu$ 和步长时不同，采取连续化策略之后，次梯度法在固定步长下收敛到了最小值。
% 注意到在 $\mu$ 减小到 |1e-2| 之前，两次优化的过程是完全相同的
% （图像不重合而是平行是由于对应的最小值不同），并且在每次 $\mu$ 减小后，函数值都有迅速的下降。
% 最终在 $1000$ 次迭代左右最终收敛，比之采取相同的连续化策略的光滑化梯度法稍慢。
%% 参考页面
% 次梯度法的算法实现参考 <LASSO_subgrad_inn.html LASSO 问题的次梯度法>，
% 连续化策略参考 <..\LASSO_con\LASSO_con.html LASSO 问题连续化策略>。
% 不使用连续化策略的次梯度法见
% <demo.html 实例：次梯度法解 LASSO 问题>。
%
% 此页面的源代码请见：
% <../download_code/lasso_subgrad/demo_cont.m demo_cont.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将
