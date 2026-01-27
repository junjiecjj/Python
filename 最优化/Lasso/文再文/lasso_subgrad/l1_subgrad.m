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
%   2016.12.20 (Haoyang Liu):
%     First version
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LASSO 问题的次梯度解法
% 对于 LASSO 问题
%
% $$ \displaystyle \min_x \frac{1}{2}\|Ax-b\|_2^2 + \mu \|x\|_1.$$
%
% 不采用连续化策略，直接对原始的正则化系数利用次梯度法求解。
%
% 注意到 $\mathrm{sign}(x)\in\partial\|x\|_1$，则次梯度法的下降方向取为
% $d^k=-\left(A^\top (Ax-b)+\mu\cdot\mathrm{sign}(x)\right)$。
%% 初始化和迭代准备
% 算法不使用连续化策略，直接对原始的 LASSO 问题进行求解。
%
% 输入信息： 要求提供数据 $A$, $b$，正则化系数 $\mu$，迭代初始点 $x^0$ 和结构体 |opts| 。
%
% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。
%
% * |out.fvec| ：每一步迭代的 LASSO 问题目标函数值
% * |out.grad_hist| ：可微部分梯度范数的历史值
% * |out.f_hist| ：目标函数的历史值
% * |out.f_hist_best| ：目标函数每一步迭代对应的历史最优值
% * |out.tt| ：运行时间
% * |out.itr| ：迭代次数
% * |out.flag| ：标记是否收敛
function [x, out] = l1_subgrad(x0, A, b, mu, opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：停机准则，当目标函数历史最优值的变化小于该值时认为满足
% * |opts.step_type| ：步长衰减的类型（见辅助函数）
% * |opts.alpha0| ：步长的初始值
% * |opts.thres| ：判断小量是否被认为为 $0$ 的阈值
if ~isfield(opts, 'maxit'); opts.maxit = 2000; end
if ~isfield(opts, 'thres'); opts.thres = 1e-4; end
if ~isfield(opts, 'step_type'); opts.step_type = 'fixed'; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'ftol'); opts.ftol = 0; end
%%%
% 初始化用于记录目标函数值、目标函数历史最优值和可微部分梯度值的矩阵。
x = x0;
out = struct();
out.f_hist = zeros(1, opts.maxit);
out.f_hist_best = zeros(1, opts.maxit);
out.g_hist = zeros(1, opts.maxit);
f_best = inf;
%% 迭代主循环
% 以 |opts.maxit| 为最大迭代次数进行迭代。
% 对于函数 $f(x)=\frac{1}{2}\|Ax-b\|^2_2+\mu\|x\|_1$，
% 用 $g=A^\top (Ax-b)$ 表示可微部分的梯度。
for k = 1:opts.maxit
    
    r = A * x - b;
    g = A' * r;
    %%%
    % 记录可微部分的梯度的范数。
    out.g_hist(k) = norm(r, 2);
    
    %%%
    % 记录当前目标函数值。
    f_now = 0.5 * norm(r, 2)^2 + mu * norm(x, 1);
    out.f_hist(k) = f_now;
    
    %%%
    % 记录当前历史最优目标函数值。
    f_best = min(f_best, f_now);
    out.f_hist_best(k) = f_best;
    %%%
    % 迭代的停止条件：当目标函数历史最优值的变化小于阈值时停止迭代。
    if k > 1 && abs(out.f_hist_best(k) - out.f_hist_best(k-1)) / abs(out.f_hist_best(1)) < opts.ftol
        break;
    end
    %%%
    % 一步次梯度法的迭代更新：函数 $f(x)$ 的次梯度可以选取为 $g+\mu\cdot\mathrm{sign}(x)$。
    x(abs(x) < opts.thres) = 0;
    sub_g = g + mu * sign(x);
    %%%
    % 利用辅助函数确定当前步步长，然后进行一步次梯度法迭代 $x^{k+1}=x^k-\alpha_k g(x^k)$。其中
    % $g(x^k)\in\partial f(x^k)$。
    alpha = set_step(k, opts);
    x = x - alpha * sub_g;
end
%%%
% 当迭代终止时，记录当前迭代步和迭代过程。
out.itr = k;
out.f_hist = out.f_hist(1:k);
out.f_hist_best = out.f_hist_best(1:k);
out.g_hist = out.g_hist(1:k);
end
%% 辅助函数
% 函数 |set_step| 在不同的设定下决定第 $k$ 步的步长。以 $\alpha_0$ 为初始步长，步长分别为
% 
% * |'fixed'|： $\alpha_k=\alpha_0$
% * |'diminishing'|： $\alpha_k=\alpha_0/\sqrt{k}$
% * |'diminishing2'|： $\alpha_k = \alpha_0/k$
function a = set_step(k, opts)
type = opts.step_type;
if strcmp(type, 'fixed')
    a = opts.alpha0;
elseif strcmp(type, 'diminishing')
    a = opts.alpha0 / sqrt(k);
elseif strcmp(type, 'diminishing2')
    a = opts.alpha0 / k;
else
    error('unsupported type.');
end
end
%% 参考页面
% 我们在 <demo.html 实例：次梯度法解 LASSO 问题> 中构建一个 LASSO
% 问题并展示该算法在其中的应用。另外，利用连续化策略的次梯度法解
% LASSO 问题，请参考 <demo_cont.html 实例：连续化次梯度法解 LASSO 问题> 。
%
% 此页面的源代码请见： <../download_code/lasso_subgrad/l1_subgrad.m l1_subgrad.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将