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

%% LASSO 问题的连续化次梯度法
% 对于 LASSO 问题
%
% $$ \displaystyle\min_x \frac{1}{2}\|Ax-b\|_2^2 + \mu \|x\|_1.$$
%
% 利用连续化策略下的次梯度法进行求解。该算法被外层连续化策略调用，完成某一固定正则化系数的内层迭代优化。
% 
% 对于目标函数，选定其次梯度为 $\displaystyle A^\top (Ax-b)+\mu\cdot \mathrm{sign}(x)$
% ，并以次梯度方向作为下降方向进行迭代。
%% 初始化和迭代准备
%%%
% 输入信息： $A$, $b$, $\mu$，迭代初始值 $x$，原问题对应的正则化系数
% $\mu_0$ ，以及提供各参数的结构体 |opts| 。
%
% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。
%
% * |out.fvec| ：每一步迭代的原始 LASSO 问题目标函数值（对应于原问题的 $\mu_0$）
% * |out.grad_hist| ：可微部分梯度范数的历史值
% * |out.fvec| ：每一步迭代的当前目标函数值（对应于当前的 $\mu_t$）
% * |out.f_hist_best| ：每一步迭代的当前目标函数历史最优值（对应于当前的 $\mu_t$）
% * |out.tt| ：运行时间
% * |out.itr| ：迭代次数
% * |out.flag| ：标记是否收敛
function [x, out] = LASSO_subgrad_inn(x, A, b, mu, mu0, opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：停机准则，当相对函数值变化或相对历史最佳函数值变化小于该值时认为满足
% * |opts.step_type| ：步长衰减的类型（见辅助函数）
% * |opts.alpha0| ：步长的初始值
% * |opts.thres| ：判断小量是否被认为为 $0$ 的阈值
if ~isfield(opts, 'maxit'); opts.maxit = 500; end
if ~isfield(opts, 'thres'); opts.thres = 1e-4; end
if ~isfield(opts, 'step_type'); opts.step_type = 'fixed'; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-12; end

%%%
% 仅在连续化外层循环的最后一步使用步长衰减。
if mu > mu0
    opts.step_type = 'fixed';
else
    opts.step_type = opts.step_type;
end

%%%
% 迭代准备，计算初始时刻 $x$ 对应的各变量值。注意到对于函数
% $f(x)=\frac{1}{2}\|Ax-b\|^2_2+\mu\|x\|_1$，可以选取其次梯度为
% $A^\top(Ax-b)+\mu \mathrm{sign}(x)$。代码中我们用 |sub_g| 表示次梯度。
tic;
out = struct();
out.fvec = [];
r = A * x - b;
gx = A' * r;
sub_g = gx + mu * sign(x);
f_best = inf;

%% 迭代主循环
% 内层迭代，以 |opts.maxit| 为最大迭代次数。
% 通过辅助函数 |set_step| 设定步长（具体的设定见辅助函数），以次梯度方向进行一步迭代。
for k = 1:opts.maxit
    alpha = set_step(k, opts);
    x = x - alpha * sub_g;
    r = A * x - b;
    g = A' * r;
    %%%
    % 将 $x$ 中绝对值小于某个设定阈值的值置为 $0$。这是由于绝对值较小的值虽然与 $0$ 接近，
    % 但次梯度却与 $0$ 处相差较大。
    x(abs(x) < opts.thres) = 0;
    sub_g = g + mu * sign(x);
    %%%
    % 记录梯度范数和当前 $x$ 对应的实际目标函数的值（当前连续化步的正则化系数 $\mu_t$
    % 对应的目标函数）以及目标函数历史值（次梯度法是非单调算法）。
    out.grad_hist(k) = norm(r, 2);
    tmp = .5*norm(r,2)^2;
    nrmx1 = norm(x,1);
    f = tmp + mu * nrmx1;
    
    out.f_hist(k) = f;
    f_best = min(f_best, f);
    out.f_hist_best(k) = f_best;
    out.fvec = [out.fvec, tmp + mu0*nrmx1];
    
    %%%
    % 详细输出模式下打印每一次迭代信息。
    if opts.verbose
        fprintf('itr: %4d \t f: %.4e \t step: %.1e\n',k, f, alpha);
    end
    %%%
    % 内层循环的停机准则，当相对函数值变化量 |FDiff| 小于设定阈值，或当循环次数大于 8 
    % 次后相对历史最佳函数值变化量 |BFDiff|
    % 小于设定阈值时，认为收敛，停止内层循环。
    FDiff = abs(out.f_hist(k) - out.f_hist(max(k-1,1))) / abs(out.f_hist_best(1));
    BFDiff = abs(out.f_hist_best(max(k - 8,1)) - min(out.f_hist_best(max(k-7,1):k)));
    if (k > 1 && FDiff < opts.ftol) || (k > 8 && BFDiff < opts.ftol)
        break;
    end
end

%%%
% 当退出循环时，向外层迭代（连续化策略）报告内层迭代的退出方式，当达到最大迭代次数退出时，
% out.flag 记为 1，否则为达到收敛，记为 0。这个指标用于判断是否进行正则化系数的衰减。
if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

%%%
% 记录输出信息。
out.itr = k;
out.tt = toc;
end
%% 辅助函数
% 函数 |set_step| 在不同的设定下决定第 $k$ 步的步长。以 $\alpha_0$ 为初始步长，步长分别为
% 
% * |'fixed'|： $\alpha_k=\alpha_0$
% * |'diminishing'|： $\alpha_k=\alpha_0/\sqrt{\hat{k}}$
% * |'diminishing2'|： $\alpha_k = \alpha_0/\hat{k}$
% 
% 其中， $\hat{k}=\max(100,k)-99$。
function a = set_step(k, opts)
type = opts.step_type;
if strcmp(type, 'fixed')
    a = opts.alpha0;
elseif strcmp(type, 'diminishing')
    a = opts.alpha0 / sqrt(max(k,100)-99);
elseif strcmp(type, 'diminishing2')
    a = opts.alpha0 / (max(k,100)-99);
else
    error('unsupported type.');
end
end
%% 参考页面
% 该函数由连续化策略调用，关于连续化策略参见 <..\LASSO_con\LASSO_con.html
% LASSO问题连续化策略>。我们在 <demo_cont.html 实例：连续化次梯度法解LASSO问题>
% 中展示该算法的一个应用。另外，不利用连续化策略的次梯度法解 LASSO 问题，
% 请参考 <l1_subgrad.html 非连续化次梯度法> 和 <demo.html 实例：次梯度法解LASSO问题>。
%
% 此页面的源代码请见：
% <../download_code/lasso_subgrad/LASSO_subgrad_inn.m LASSO_subgrad_inn.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将