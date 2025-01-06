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


%% 实例：罚函数法解基追踪问题
% 考虑基追踪问题：
%
% $$ \min\|x\|_1,\quad \mathrm{s.t.}\quad Ax=b,$$
% 
% 其为一个线性等式约束的非光滑优化问题，使用二次罚函数作用于等式约束则得到 LASSO 问题：
% 
% $$\min \|x\|_1+\frac{\sigma}{2}\|Ax-b\|^2, $$
% 其中 $\sigma$ 为罚因子。 
% 
% 令 $\mu = \frac{1}{\sigma}$，罚因子逐渐增加到无穷，则等价于 LASSO 问题中的 $\mu$
% 逐渐减小，对应于 LASSO 问题求解的连续化策略。当 $\mu$ 趋于 $0$ 时 LASSO 
% 问题的解收敛到 BP 问题的解。这里我们设定一个固定的 $\mu$（其越小逼近基追踪问题的质量越高）为目标，从一个较大的 $\mu$
% 逐渐减小到该值。
%% 生成 LASSO 问题
clear;
%%%
% 设定随机种子。
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%%
% 构造 LASSO 优化问题
% 
% $$ \min_x f(x) :=\frac{1}{2}\|Ax-b\|_2^2 + \mu\|x\|_1, $$
% 
% 生成随机的矩阵 $A$ 和向量 $u$ 以使得 $b=Au$。
m = 512;
n = 1024; 
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
x0 = zeros(n, 1);

%% 罚函数法
% 罚函数法在这里对应于 LASSO 问题的连续化策略，我们从 $\mu=10$ 开始以 $\gamma = 10^{-1}$
% 的比例衰减到 $\mu=10^{-3}$，以逐渐近似求解原始的基追踪问题。对于每个 $\mu$对应的 LASSO 子问题，使用固定步长的次梯度法进行求解。
opts = struct();
opts.maxit = 600;
opts.alpha0 = 0.0002; 
%%%
% 固定步长的次梯度法求解子问题。 |f_val| 记录求解过程中 $\frac{f(x)}{\mu}$ 的值， |g_val| 记录约束违反度 $\frac{\|Ax^k-b\|}{\|b\|}$ 的值。
opts.step_type = 'fixed';
x = x0;
f_val = [];
g_val = [];
for mu = [10, 1, 1e-1, 1e-2, 1e-3]
    addpath('../lasso_subgrad');
    [x, out] = l1_subgrad(x, A, b, mu, opts);
    f_val = [f_val, out.f_hist_best(1:out.itr) / mu];
    g_val = [g_val, out.g_hist(1:out.itr) / norm(b)];
end

%% 次梯度法解 LASSO 问题
% 不适用连续化策略，直接对 $\mu=10^{-3}$时的 LASSO 问题利用次梯度法进行求解。将结果与罚函数法（LASSO 的连续化策略）
% 的求解效果进行对比。
opts.maxit = numel(f_val);
opts.ftol = 0;
[x1, out] = l1_subgrad(x0, A, b, 1e-3, opts);
f_val1 = out.f_hist_best / 1e-3;
g_val1 = out.g_hist / norm(b);

%% 结果可视化
fig1 = figure;
x = 1:numel(f_val);
semilogy(x, f_val, 'k-', x, f_val1, 'k--');
legend('罚函数法', '次梯度法');
ylabel('$\hat{f}^k/\mu$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig1, '-depsc','penalty1.eps');

fig2= figure;
semilogy(x, g_val, 'k-', x, g_val1, 'k--');
legend('罚函数法', '次梯度法');
ylabel('$\|Ax^k - b\|/\|b\|$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig2, '-depsc','penalty2.eps');

%% 结果分析
% 从函数值和约束的违反度两个角度，罚函数法的效果均好于直接使用次梯度法。
% 在迭代初期，由于 $\mu$较大，这意味着约束 $Ax = b$可以不满足，
% 从而算法可将优化的重点放到 $\|x\|_1$上；随着迭代进行， $\mu$ 单调减小，
% 此时算法将更注重于可行性( $\|Ax -b\|_2$的大小)。直接取 $\mu = 10^{-3}$效果不佳，
% 这是因为惩罚项 $\|Ax -b\|_2^2$的权重太大，次梯度法会尽量使得迭代点满足 $Ax = b$，而忽视了 $\|x\|_1$项的作用。

%% 参考页面
% 关于固定正则化系数 $\lambda$ 的 LASSO 问题次梯度法，请参考
% <../lasso_subgrad/l1_subgrad.html LASSO 问题的次梯度解法>。
%
% 另外，LASSO 的连续化策略，请参考 
% <../LASSO_con/LASSO_con.html LASSO 的连续化策略>。
%
% 此页面的源代码请见： <../download_code/pm_bp/demo_cont.m demo_cont.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将