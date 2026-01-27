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


%% 实例：次梯度法解 LASSO 问题
% 考虑 LASSO 问题
%
% $$ \displaystyle \min_x \frac{1}{2}\|Ax-b\|_2^2+\mu\|x\|_1. $$
%
% 这里，利用固定正则化系数 $\mu$ 的次梯度法进行求解，比较不同步长的效果。
%% 构造 LASSO 优化问题
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
% 生成随机的矩阵 $A$ 和向量 $u$ 以使得 $b=Au$。
m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
%%%
% 实验选取固定的正则化系数 $\mu=1$。随机迭代初始点。
mu = 1;
x0 = u + 1e-1 * randn(n, 1);

%% 求解 LASSO 优化问题
%%%
% 首先在更严格的停机准则下进行试验，将收敛时得到的历史最优函数值作为真实的最优值的参考 $f^*$。
opts_g = struct();
opts_g.maxit = 50000;
opts_g.alpha0 = 1e-6;
opts_g.step_type = 'fixed';
[x_g, out_g] = l1_subgrad(u, A, b, mu, opts_g);
f_star = out_g.f_hist_best(end);
%%%
% 取步长为 $0.0005$ 进行试验。
opts = struct();
opts.maxit = 3000;
opts.alpha0 = 0.0005;
opts.step_type = 'fixed';
[x, out] = l1_subgrad(x0, A, b, mu, opts);
data1 = (out.f_hist - f_star) / f_star;
data1_best = (out.f_hist_best - f_star) / f_star;
%%%
% 取步长为 $0.0002$ 进行试验。
opts.alpha0 = 0.0002;
[x, out] = l1_subgrad(x0, A, b, mu, opts);
data2 = (out.f_hist - f_star) / f_star;
data2_best = (out.f_hist_best - f_star) / f_star;
%%%
% 取步长为 $0.0001$ 进行试验。
opts.alpha0 = 0.0001;
[x, out] = l1_subgrad(x0, A, b, mu, opts);
data3 = (out.f_hist - f_star) / f_star;
data3_best = (out.f_hist_best - f_star) / f_star;
%%%
% 取步长为 $0.002$ 开始的消失步长 $0.002/\sqrt{k}$ 进行试验。
opts.step_type = 'diminishing';
opts.alpha0 = 0.002;
[x, out] = l1_subgrad(x0, A, b, mu, opts);
data4 = (out.f_hist - f_star) / f_star;
data4_best = (out.f_hist_best - f_star) / f_star;

%% 结果可视化
% 首先可视化每一步迭代得到的目标函数值的相对误差，即 ${\left(f(x^k)-f^*\right)}/{f^*}$。
fig = figure;
semilogy(0:length(data1)-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:length(data2)-1, data2, '--','Color',[0.99 0.1 0.2], 'LineWidth',1.2);
hold on
semilogy(0:length(data3)-1, data3, '-.','Color',[0.99 0.1 0.99], 'LineWidth',1.5);
hold on
semilogy(0:length(data4)-1, data4, ':','Color',[0.5 0.2 0.1], 'LineWidth',1.8);
legend('0.0005', '0.0002', '0.0001', '消失步长');
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','f.eps');

%%%
% 展示第 $k$ 步迭代的历史最优目标函数值的相对误差，即 ${\left(\hat{f}(x^k)-f^*\right)}/{f^*}$。
fig = figure;
semilogy(0:length(data1_best)-1, data1_best, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:length(data2_best)-1, data2_best, '--','Color',[0.99 0.1 0.2], 'LineWidth',1.2);
hold on
semilogy(0:length(data3_best)-1, data3_best, '-.','Color',[0.99 0.1 0.99], 'LineWidth',1.5);
hold on
semilogy(0:length(data4_best)-1, data4_best, ':','Color',[0.5 0.2 0.1], 'LineWidth',1.8);
legend('0.0005', '0.0002', '0.0001', '消失步长');
ylabel('$(\hat{f}(x^k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','f_best.eps');
%% 结果分析
% 上图展示了在不使用连续化策略的情况下，不同步长的次梯度法对 LASSO 问题的收敛性。
% 不同的固定步长会在 $3000$ 步内均达到次优解，且步长越小得到的次优解越小，并未收敛到最优值。
% 而对于消失步长，算法最终收敛。
%
% 另外，从左图中我们看出，次梯度法本身是非单调方法，在收敛过程之中目标函数值或有上升。
%% 参考页面
% 次梯度法解 LASSO 问题的算法详见 <l1_subgrad.html LASSO 问题的非连续化次梯度法>
% ，使用连续化策略的次梯度法见 <demo_cont.html 实例：连续化梯度法解 LASSO 问题>。
%
% 此页面的源代码请见： <../download_code/lasso_subgrad/demo.m demo.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将