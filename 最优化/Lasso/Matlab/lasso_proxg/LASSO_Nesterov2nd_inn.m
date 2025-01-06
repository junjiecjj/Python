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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LASSO 问题的第二类 Nesterov 加速算法
% 对于 LASSO 问题
%
% $$ \displaystyle\min_x \frac{1}{2}\|Ax-b\|_2^2 + \mu \|x\|_1,$$
%
% 利用第二类 Nesterov 加速的近似点梯度法进行优化。
%
% 该算法被外层连续化策略调用，在连续化策略下完成某一固定正则化系数的内层迭代优化。
% 第二类 Nesterov 加速算法的迭代格式如下
%
% $$ \begin{array}{rl}\displaystyle
% z^k= &\hspace{-0.5em} (1-\gamma_k)x^{k-1}+\gamma_k y^{k-1}, \\
% y^k= &\hspace{-0.5em} \mathrm{prox}_{(t_k/\gamma_k)h}
% \left(y^{k-1}-\frac{t_k}{\gamma_k}A^\top(Az^k-b)\right), \\
% x^k= &\hspace{-0.5em} (1-\gamma_k)x^{k-1}+\gamma_k y^k.
% \end{array} $$
%
%% 初始化和迭代准备
% 函数在 LASSO 连续化策略下，完成内层迭代的优化。
%
% 输入信息： $A$, $b$, $\mu$ ，迭代初始值 $x^0$ ，原问题对应的正则化系数 $\mu_0$ ，
% 以及提供各参数的结构体 |opts| 。
%
% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。
%
% * |out.fvec| ：每一步迭代的原始 LASSO 问题目标函数值（对应于原问题的 $\mu_0$）
% * |out.fval| ：迭代终止时的原始 LASSO 问题目标函数值（对应于原问题的 $\mu_0$）
% * |out.nrmG| ：迭代终止时的梯度范数
% * |out.tt| ：运行时间
% * |out.itr| ：迭代次数
% * |out.flag| ：记录是否收敛
function [x, out] = LASSO_Nesterov2nd_inn(x0, A, b, mu, mu0, opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：针对函数值的停机准则，当相邻两次迭代函数值之差小于该值时认为该条件满足
% * |opts.gtol| ：针对梯度的停机准则，当当前步梯度范数小于该值时认为该条件满足
% * |opts.alpha0| ：步长的初始值
% * |optsz.verbose| ：不为 0 时输出每步迭代信息，否则不输出
if ~isfield(opts, 'maxit'); opts.maxit = 10000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1e-3; end
%%%
% 初始化， $t$ 为步长，初始步长由 |opts.alpha0| 提供。
k = 0;
tt = tic;
x = x0;
y = x;
t = opts.alpha0;
%%%
% $g=A^\top(Ax-b)$ 为可微部分的梯度，
% $f=\frac{1}{2}\|Ax-b\|^2+\mu\|x\|_1$ 为优化的目标函数, |nrmG|
% 在初始时刻用一步近似点梯度法（步长为 $1$）的位移作为梯度的估计，用于收敛性的判断。
fp = inf;
r = A*x0 - b;
g = A'*r;
tmp = .5*norm(r,2)^2;
f = tmp + mu0*norm(x,1);
nrmG = norm(x - prox(x - g,mu),2);
out = struct();
out.fvec = tmp + mu0*norm(x,1);
%% 迭代主循环
% 当达到最大迭代次数，或梯度或函数值的变化大于阈值时，退出迭代。
while k < opts.maxit && nrmG > opts.gtol && abs(f - fp) > opts.ftol
    fp = f;
    %%%
    % 第二类 Nesterov 加速算法迭代。定义 $\gamma_k=\frac{2}{k+1}$。
    % 记 $\phi(z)=\frac{1}{2}\|Az-b\|_2^2$, $\nabla \phi(z^k)=A^\top(Az^k-b)$。
    %
    % $$ \begin{array}{rl}
    % \displaystyle z^k = & \hspace{-0.5em}(1-\gamma_k)x^{k-1}+\gamma_ky^{k-1},\\
    % \displaystyle y^k = & \hspace{-0.5em}
    % \mathrm{prox}_{(t_k/\gamma_k)h}(y^{k-1}-\frac{t_k}{\gamma_k}\nabla \phi(z^k)),\\
    % x^k= & \hspace{-0.5em}(1-\gamma_k)x^{k-1}+\gamma_ky^k.
    % \end{array} $$
    %
    % 通过迭代更新 $\{x^k\}\{y^k\}\{z^k\}$ 三个序列实现第二类 Nesterov 加速算法。
    rk = 2/(k+2);
    z = (1- rk)*x + rk*y;
    r = A * z - b;
    g = A' * r;
    y = prox(y - t/rk * g, t/rk*mu);
    x = (1 - rk)*x + rk*y;
    
    %%%
    % 更新变量和函数值。
    Axb = A*x - b;
    nrmG = norm(x - prox(x - A'*(A*x -b), mu),2);
    f = .5*norm(Axb,2)^2 + mu0*norm(x,1);
    
    %%%
    % 迭代步加一，记录当前函数值。输出信息。
    k = k + 1;
    out.fvec= [out.fvec, f];
    
    if opts.verbose
        fprintf('itr: %d\tt: %e\tfval: %e\tnrmG: %e\n', k, t, f, nrmG);
    end
    %%%
    % 特别的，除了上述的停机准则外，如果连续 $20$ 步的函数值不下降，则停止内层循环。
    if k > 20 && min(out.fvec(k-19:k)) > out.fvec(k-20)
        break;
    end
end

%%%
% 当达到最大迭代次数退出时， out.flag 记为 1 ，否则为达到收敛，记为 0。
% 这个指标用于判断是否进行正则化系数的衰减。
if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

%%%
% 记录输出信息。
out.fvec = out.fvec(1:k);
out.fval = f;
out.itr = k;
out.tt = toc(tt);
out.nrmG = nrmG;
end
%% 辅助函数
% 函数 $h(x)=\mu\|x\|_1$ 对应的邻近算子 $\mathrm{sign}(x)\max\{|x|-\mu,0\}$。
function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end
%% 参考页面
% 该函数由连续化策略调用，关于连续化策略参见
% <..\LASSO_con\LASSO_con.html LASSO问题连续化策略>。
%
% 在页面 <demo_proxg.html 实例：近似点梯度法和 Nesterov 加速算法求解 LASSO 问题>
% 我们展示该算法的应用。另外，参考
% <LASSO_proximal_grad_inn.html LASSO 问题的近似点梯度法>、
% <LASSO_Nesterov_inn.html LASSO 问题的 Nesterov 加速算法>。
%
% 此页面的源代码请见：
% <../download_code/lasso_proxg/LASSO_Nesterov2nd_inn.m
% LASSO_Nesterov2nd_inn.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将