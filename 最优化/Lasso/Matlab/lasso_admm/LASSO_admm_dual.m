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
%   2020.2.15 (Jiang Hu):
%     First version
%
%   2020.7.16 (Jiang Hu):
%     Code improving 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 利用交替方向乘子法 (ADMM) 求解 LASSO 对偶问题
% 交替方向乘子法，即 the Alternating Direction Method of Multipliers
% （ADMM），利用 ADMM 求解 LASSO 问题的对偶问题。
%
% 针对 LASSO 问题
%
% $$ \min_{x,z} \frac{1}{2} \|Ax-b\|_2^2 + \mu \|x\|_1,$$
%
% 考虑其对偶问题的 ADMM 等价形式
%
% $$ \begin{array}{rl}
% \displaystyle\min_y &\hspace{-0.5em} b^\top y +\frac{1}{2}\|y\|_2^2+I_{\|z\|_\infty\le\mu}(z),\\
% \displaystyle\mathrm{s.t.} &\hspace{-0.5em} A^\top y + z = 0.
% \end{array} $$
%
% 引入 $x$ 作为拉格朗日乘子，得到增广拉格朗日函数
% $L_\rho(y,z,x)=b^\top y+\frac{1}{2}\|y\|^2+I_{\|z\|_\infty\le\mu}(z)
% -x^\top(A^\top y+z)+\frac{\rho}{2}\|A^\top y+z\|^2$。
% 在 ADMM 的每一步迭代中，交替更新 $y$, $z$，在更新 $y$( $z$) 的时候 $z$( $y$) 固定（看成常量）。
%% 初始化和迭代准备
% 函数通过优化上面给出的增广拉格朗日函数，以得到 LASSO 问题的解。
%
% 输入信息： $A$, $b$, $\mu$ ，迭代初始值 $x^0$ 以及提供各参数的结构体 |opts| .
%
% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。
%
% * |out.fvec| ：每一步迭代的 LASSO 问题目标函数值
% * |out.fval| ：迭代终止时的 LASSO 问题目标函数值
% * |out.tt| ：运行时间
% * |out.itr| ：迭代次数
% * |out.y| ：迭代终止时的对偶变量 $y$ 的值
% * |out.nrmC| ：迭代终止时的约束违反度，在一定程度上反映收敛性
function [x, out] = LASSO_admm_dual(x0, A, b, mu, opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：针对函数值的停机准则，当相邻两次迭代函数值之差小于该值时认为该条件满足
% * |opts.gtol| ：针对 $x$ 的梯度的停机准则，当当前步的梯度范数小于该值时认为该条件满足
% * |opts.sigma| ：增广拉格朗日系数
% * |opts.gamma| ： $x$ 更新的步长
% * |opts.verbose| ：不为 0 时输出每步迭代信息，否则不输出
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'sigma'); opts.sigma = 0.5; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end

%%%
% 迭代准备。
tt = tic;
k = 0;
x = x0;
out = struct();
%%%
% 初始化对偶问题的对偶变量 $y$。
[m, ~] = size(A);
sm = opts.sigma;
y = zeros(m,1);

%%%
% 记录在初始时刻原问题目标函数值。
f = .5*norm(A*x - b,2)^2 + mu*norm(x,1);
fp = inf;
out.fvec = f;
nrmC = inf;

%%%
% Cholesky 分解， $R$ 为上三角矩阵且 $R^\top R=A^\top A + \sigma I_m$。
% 与原始问题同样的，由于罚因子在算法的迭代过程中未变化，事先缓存 Cholesky 分解可以加速迭代过程。
W = eye(m) + sm * (A * A');
R = chol(W);

%% 迭代主循环
% 迭代主循环，当 (1) 达到最大迭代次数或 (2) 目标函数的变化小于阈值或
% (3) 自变量 $x$ 的变化量小于阈值时，退出迭代循环。
while k < opts.maxit && abs(f - fp) > opts.ftol && nrmC > opts.gtol
    fp = f;
    %%%
    % 对 $z$ 的更新为向无穷范数球做欧式投影，
    % $\displaystyle z^{k+1}=\mathcal{P}_{\|\cdot\|_\infty\le\mu}
    % \left(x^k/\sigma-A^\top y^k\right)$。
    z = proj( - A' * y + x / sm, mu);
    %%%
    % 针对 $y$ 的子问题，即为求解线性方程组 $(I+\sigma AA^\top)y^{k+1}=A(x^k-\sigma
    % z^{k+1})-b$。
    % 这里同样利用了事先缓存的 Cholesky 分解来加速 $y^{k+1}$ 的计算。
    h = A * (- z*sm + x) - b;
    y = R \ (R' \ h);
    %%%
    % 令 $c^{k+1}=A^\top y^{k+1} + z^{k+1}$ 为等式约束的约束违反度。
    % 增广拉格朗日函数对 $x$ 的梯度为 $\frac{\partial L_\rho(y,z,x)}{\partial x}=-\sigma
    % c$。
    % 针对 $x$ 的子问题，进行一步梯度上升， $x^{k+1}=x^k - \gamma\sigma (A^\top y^{k+1} +
    % z^{k+1})$。利用 |nrmC| （约束违反度的范数）作为停机判断依据。
    c = z + A' * y;
    x = x - opts.gamma * sm * c;
    nrmC = norm(c,2);
    %%%
    % 计算更新后的目标函数值，记录在 |out.fvec| 中。当 |opts.verbose| 不为 0 时输出详细的迭代信息。
    f = .5*norm(A*x - b,2)^2 + mu*norm(x,1);
    if opts.verbose
        fprintf('itr: %4d\tfval: %e\tfeasi: %.1e\n', k, f, nrmC);
    end
    k = k + 1;
    out.fvec = [out.fvec; f];
end
%%%
% 记录输出信息。
out.y = y;
out.fval = f;
out.itr = k;
out.tt = toc(tt);
out.nrmC = nrmC;
end
%% 辅助函数
% 到无穷范数球 $\{x\big| \|x\|_\infty \le t\}$ 的投影函数。
function w = proj(x, t)
w = min(t, max(x, -t));
end
%% 参考页面
% 在页面 <demo_admm.html 实例：交替方向乘子法解 LASSO 问题> 中我们展示此算法的一个应用。
% 另外，对 LASSO 原问题的
% ADMM 参考页面 <LASSO_admm_primal.html 利用交替方向乘子法求解 LASSO 原问题>。
%
% 此页面的源代码请见： <../download_code/lasso_admm/LASSO_admm_dual.m
% LASSO_admm_dual.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将