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
%   2020.2.15 (Jiang Hu):
%     Parameter tuning
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LASSO 问题的分块坐标下降法
% 利用分块坐标下降法（BCD）优化如下的 LASSO 问题
%
% $$ \displaystyle\min_x \frac{1}{2}\|Ax-b\|_2^2 + \mu \|x\|_1.$$
%
% 该算法被外层连续化策略函数调用，完成对某一固定正则化系数 $\mu_k$ 的内层迭代优化。
%
% 分块坐标下降法将 $x$ 的第 $i$ 个分量 $x_i$ 作为第 $i$ 块变量。记 $a_i$ 为 $A$ 的第 $i$ 列，
% $\bar{x}_i$ 和 $\bar{A}_i$ 为 $x$ 去掉第 $i$ 个分量和 $A$ 去掉第 $i$ 列后的量，
% 则在第 $i$ 块的更新中，原问题等价于
%
% $$ \displaystyle\min_{x_i}\mu|x_i|+\mu\|\bar{x}_i\|_1
% +\frac{1}{2}\|a_ix_i-(b-\bar{A}_i\bar{x}_i)\|_2^2. $$
%
% 记 $c_i=b-\bar{A}_i\bar{x}_i$ ，则由上述问题可以得到迭代格式
%
% $$ \displaystyle x_i^k=
% \left\{\begin{array}{ll}
% \displaystyle\frac{a_i^\top c_i-\mu}{\|a_i\|^2_2}, & a_i^\top c_i>\mu;\\
% \displaystyle\frac{a_i^\top c_i+\mu}{\|a_i\|^2_2}, & a_i^\top c_i<-\mu;\\
% 0, & \mathrm{otherwise}.
% \end{array}\right. $$
%% 初始化和迭代准备
% 函数在 LASSO 连续化策略下，完成内层迭代的优化。
% 
% 输入信息： $A$, $b$，当前内层迭代的正则化系数
% $\mu$ ，迭代初始值 $x^0$ ，原问题对应的正则化系数 
% $\mu_0$ ，以及提供各参数的结构体 |opts| 。
% 
% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。
%
% * |out.fvec| ：每一步迭代的原始目标函数值（对应于原问题的 $\mu_0$）
% * |out.fval| ：迭代终止时的原始目标函数值（对应于原问题的 $\mu_0$）
% * |out.nrmG| ：迭代终止时的梯度范数
% * |out.tt| ：运行时间
% * |out.itr| ：迭代次数
% * |out.flag| ：标记是否达到收敛
function [x, out] = LASSO_bcd_inn(x0, A, b, mu, mu0, opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：针对函数值的停机准则，当相邻两次迭代函数值之差小于该值时认为该条件满足
% * |opts.gtol| ：针对梯度的停机准则，当当前步梯度范数小于该值时认为该条件满足
% * |opts.verbose| ：不为 0 时输出每步迭代信息，否则不输出
if ~isfield(opts, 'maxit'); opts.maxit = 200; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
%%%
% 迭代准备，计算初始时刻 $x$ 对应的各变量值。
out = struct();
x = x0;
tt = tic;
[m,n] = size(A);
Ax = A*x;
r = Ax - b;
f = .5*norm(r,2)^2 + mu0*norm(x,1);

%%%
% 记录初始时刻的目标函数值（对应原正则化系数 mu_0）。
out.fvec = f;

%% 迭代主循环
% 内层迭代，以 |maxit| 为最大迭代次数。
for k = 1:opts.maxit
    %%%
    % 记录上一步的迭代结果。
    fp = f;
    %%%
    % BCD 的迭代过程，每一步迭代分别针对 $x$ 的第 $i$ 个坐标分量进行优化，计算
    % $c_i=b-\bar{A}_i\bar{x}_i$。
    for i = 1:n
        a = A(:,i);
        Axnew = Ax - a*x(i);
        c = b - Axnew;
        %%%
        % BCD 的每一个坐标块的下降。
        %
        % $$ \displaystyle x_i^k=
        % \left\{\begin{array}{ll}
        % \displaystyle\frac{a_i^\top c_i-\mu}{\|a_i\|^2_2}, & a_i^\top c_i>\mu;\\
        % \displaystyle\frac{a_i^\top c_i+\mu}{\|a_i\|^2_5}, & a_i^\top c_i<-\mu;\\
        % 0, & \mathrm{otherwise}.
        % \end{array}\right. $$
        ac = a'*c;
        if ac > mu
            x(i) = (ac - mu)/norm(a,2)^2;
        elseif ac < -mu
            x(i) = (ac + mu)/norm(a,2)^2;
        else
            x(i) = 0;
        end
        
        %%%
        % $Ax=\bar{A}_i\bar{x}_i+a_ix_i.$
        Ax = Axnew + a*x(i);
    end
    %%%
    % 从上面的循环中退出， $x$ 的全部 $n$ 个坐标分量依次优化完成，更新并记录相关变量的值。
    Axb = Ax - b;
    f = .5*norm(Axb,2)^2 + mu0*norm(x,1);
    
    %%%
    % 利用一步近似点梯度法作为梯度的近似。
    nrmG = norm(x - prox(x - A'*Axb, mu));
    %%%
    % 记录一步 BCD 迭代的函数值。
    out.fvec = [out.fvec, f];
    %%%
    % 当 verbose 不为 0 时，输出迭代信息。
    if opts.verbose
        fprintf('itr: %d\t fval: %e \t nrmG: %.1e \n', k, f, nrmG);
    end
    
    %%%
    % 停机准则，当函数值变化小于阈值或者梯度范数小于阈值时，认为达到收敛，退出迭代循环。
    % 当退出循环时，向外层迭代（连续化策略）报告内层迭代的退出方式，当达到最大迭代次数退出时，
    % |out.flag| 记为 1，否则则为达到收敛，记为 0。 这个指标用于判断是否进行正则化系数的衰减。
    if abs(f-fp) < opts.ftol || nrmG < opts.gtol
        break;
    end
end

if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

%%%
% 记录内层迭代的输出。
out.fvec = out.fvec(1:k);
out.fval = f;
out.itr = k;
out.tt = toc(tt);
end
%% 辅助函数
% 函数 $h(x)=\mu\|x\|_1$ 对应的邻近算子 $\mathrm{sign}(x)\max\{|x|-\mu,0\}$。
function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end
%% 参考页面
% 该函数由连续化策略调用，关于连续化策略参见 <..\LASSO_con\LASSO_con.html LASSO问题连续化策略>。
%
% 我们在页面 <demo_bcd.html 实例：利用分块坐标下降法求解 LASSO 问题>
% 中构建一个 LASSO 问题并展示该算法的一个应用。
%
% 此页面的源代码请见： <../download_code/lasso_bcd/LASSO_bcd_inn.m LASSO_bcd_inn.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将