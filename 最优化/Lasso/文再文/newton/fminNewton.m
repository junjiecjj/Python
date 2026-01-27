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
%   2020.4.18 (Jiang Hu):
%     First version
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 利用牛顿-共轭梯度法解优化问题
% 该算法利用非精确牛顿法（牛顿-共轭梯度法）求解无约束优化问题：
%
% $$ \min_{x} \;\; f(x)$$
%
% 在第 $k$ 步迭代，下降方向 $d^k$ 通过求解下面的牛顿方程 
% $\displaystyle (\nabla^2 f(x^k)) d^k = -\nabla f(x^k)$
% 得到。选取合适的步长 $\alpha_k$，牛顿法的迭代格式为
%
% $$ x^{k+1} = x^{k} + \alpha_k d_k.$$
%
% 对于规模较大的问题，精确求解牛顿方程组的代价比较高。事实上，牛顿方程求解等价于无约束二次优化问题：
%
% $$ \displaystyle\min_{d^k} \frac{1}{2}(d^k)^\top \nabla^2 f(x^k) d^k +
% (\nabla f(x^k))^\top d^k, $$
%
% 其可以通过共轭梯度法来进行求解。
%
%% 初始化和迭代准备
% 输入信息：迭代初始点 |x| ，目标函数 |fun| （依次返回给定 |x| 处的函数值和梯度），
% 海瑟矩阵 |hess| 以及提供参数的结构体 |opts| 。
%
% 输出信息：迭代得到的解 |x| 和迭代信息结构体 |out| 。
%
% * |out.msg| ：标记是否达到收敛
% * |out.nrmG| ：迭代退出时的梯度范数
% * |out.iter| ：迭代退出时的迭代步数、函数值
% * |out.f| ：迭代退出时的目标函数值
% * |out.nfe| ：调用原函数的次数
function [x, out] = fminNewton(x, fun, hess, opts, varargin)
%%%
% 检查输入信息，MATLAB 以 |nargin| 表示函数输入的参数个数。当参数量不足 3 时报错，
% 等于 3 时认为 |opts| 为空结构体，即全部采用默认参数。
%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.xtol| ：针对优化变量的停机准则
% * |opts.gtol| ：针对梯度范数的停机准则
% * |opts.ftol| ：针对函数值的停机准则
% * |opts.rho1| ：线搜索参数 $\rho_1$
% * |opts.rho2| ：线搜索参数 $\rho_2$
% * |opts.maxit| ：主循环的最大迭代次数
% * |opts.verbose| ： |>=1| 时输出迭代信息
% * |opts.itPrint| ：每隔几步输出一次迭代信息
if (nargin < 3); error('[x, out] = fminNewton(x, fun, hess, opts)'); end
if (nargin < 4); opts = []; end
if ~isfield(opts,'gtol');     opts.gtol = 1e-6;   end
if ~isfield(opts,'xtol');     opts.xtol = 1e-6;   end
if ~isfield(opts,'ftol');     opts.ftol = 1e-12;  end
if ~isfield(opts, 'rho1');      opts.rho1  = 1e-4; end
if ~isfield(opts, 'rho2');      opts.rho2  = 0.9; end
if ~isfield(opts,'maxit');    opts.maxit = 200;   end
if ~isfield(opts,'verbose');   opts.verbose = 0;    end
if ~isfield(opts,'itPrint');    opts.itPrint = 1;   end
%%%
% 从结构体 |opts| 中复制参数。
maxit   = opts.maxit;   verbose  = opts.verbose;  itPrint = opts.itPrint;
xtol    = opts.xtol;    gtol    = opts.gtol;    ftol    = opts.ftol;
%%%
% |parsls| 为由线搜索参数构成的结构体。
parsls.ftol = opts.rho1;  parsls.gtol = opts.rho2;
%%%
% 迭代准备，计算初始点 $x$ 处的函数值和梯度值。
[f,g] = fun(x);
nrmg = norm(g,2);
nrmx = norm(x,2);
%%%
% 预先设定输出信息，在没有达到收敛条件时，记为超出最大迭代次数退出。
out = struct();
out.msg = 'MaxIter';
out.nfe = 1;
out.nrmG = nrmg;
%%%
% 当需要详细输出时，设定输出格式。
if verbose >= 1
    if ispc; str1 = '  %10s'; str2 = '  %8s';
    else     str1 = '  %10s'; str2 = '  %8s'; end
    stra = ['%5s', str1, str2, str2, str2, '\n'];
    str_head = sprintf(stra, ...
        'iter', 'F', 'nrmg', 'fdiff', 'xdiff');
    fprintf('%s', str_head);
    str_num = ['%4d  %+8.7e  %+2.1e  %+2.1e  %+2.1e\n'];
end
%%%
% 非精确牛顿法使用共轭梯度法求解牛顿方程， |opts_tCG| 为共轭梯度法提供参数。
opts_tCG = struct();

%% 迭代主循环
for iter = 1:maxit
    %%%
    % 记录上一步迭代的信息（上一步的优化变量和对应的目标函数值、梯度、范数）。
    fp = f;
    gp = g;
    xp = x;
    nrmxp = nrmx;
    %%%
    % 调用截断共轭梯度法 |tCG| （见参考页面）不精确地求解牛顿方程得到牛顿方向 $d$。
    % 这里函数 |hess(x,d)| 对应矩阵向量乘 $\nabla^2 f(x) d$。
    opts_tCG.kappa = 0.1;
    opts_tCG.theta = 1;
    hess_tCG = @(d) hess(x,d);
    [d, ~, ~] = tCG(x, gp, hess_tCG, inf, opts_tCG);
    %%%
    % 沿方向 $d$ 做线搜索。调用函数 |ls_csrch| 进行线搜索，具体步骤可以参考
    % MINPACK-2。
    %
    % 首先初始化线搜索标记 |workls.task| 为 1， |deriv| 为当前线搜索方向上的导数。
    % 线搜索函数寻找合适的步长 $\alpha_k$，在对应的搜索方向上满足条件
    %
    % $$ \begin{array}{rl}
    % f(x^k+\alpha_k d^k)&\hspace{-0.5em}\le
    % f(x^k)+\rho_1\cdot\alpha_kg(x^k), \\
    % |g(x^k+\alpha_kd^k)|&\hspace{-0.5em}\le \rho_2\cdot |g(x^k)|.
    % \end{array} $$
    %
    % |ls_csrch| 每次调用只执行线搜索的一步，并用 |workls.task| 指示下一步应当执行的操作。
    % 此处 |workls.task==2| 意味着需要重新计算当前点函数值和梯度等。具体的步长寻找过程可以参考源文件。
    %
    % 当线搜索条件满足时，退出线搜索循环，得到更新的迭代点 $x$。
    workls.task = 1;
    deriv = d'*g;
    normd = norm(d);
    
    stp = 1;
    while 1
        [stp, f, deriv, parsls, workls] = ....
            ls_csrch(stp, f, deriv , parsls , workls);
        if (workls.task == 2)
            x = xp + stp*d;
            [f,  g] = feval(fun, x, varargin{:});
            out.nfe = out.nfe + 1;
            deriv = g'*d;
        else
            break
        end
    end
    %%%
    % 计算一步更新后的点 $x$处的相关数据以判断是否停机。
    % |nrms| 表示 $\|x^{k+1}-x^k\|_2$，
    % |xdiff| 表示 $\|x^{k+1}-x^k\|/\max\{1,\|x^k\|\}$ 为迭代点 $x$的相对变化量，
    % |nrmg| 表示 $x^k$ 处梯度范数， |nrmx| 为点 $x$ 的范数，
    % |fdiff| 为函数值相对变化量。 |out.nfe| 记录函数值计算的总次数。
    nrms = stp*normd;
    xdiff = nrms/max(nrmxp,1);
    nrmg = norm(g,2);
    out.nrmG = [out.nrmG; nrmg];
    nrmx = norm(x,2);
    out.nfe = out.nfe + 1;
    fdiff = abs(fp-f)/(abs(fp)+1);
    %%%
    % 停机判断：当梯度范数小于阈值，或者函数值的相对变化量和优化变量的相对变化量均小于阈值时，
    % 迭代终止。
    cstop = nrmg <= gtol || (abs(fdiff) <= ftol && abs(xdiff) <= xtol);
    %%%
    % 当需要详细输出时，在（1）开始迭代时（2）达到收敛时（3）达到最大迭代次数退出迭代时
    % （4）每若干步，打印详细结果。
    % 当满足收敛条件时，标记为达到最优值。退出迭代，记录退出信息为达到最优值退出。
    if verbose>=1 && (cstop || iter == 1 || iter==maxit || mod(iter,itPrint)==0)
        if mod(iter,20*itPrint) == 0 && iter ~= maxit && ~cstop
            fprintf('\n%s', str_head);
        end
        
        fprintf(str_num, ...
            iter, f, nrmg, fdiff, xdiff);
    end
    if cstop
        out.msg = 'Optimal';
        break;
    end
end
%%%
% 迭代退出时，记录输出信息。
out.iter = iter;
out.f    = f;
out.nrmg = nrmg;
end
%% 参考页面
% 算法中的共轭梯度法部分，详见 <tCG.html 共轭梯度法> 。在 <demo_lr_Newton.html
% 实例：应用牛顿-共轭梯度法解逻辑回归问题>中，我们展示该算法的一个应用。
%
% 代码中利用了翻译为 MATLAB 代码的 MINPACK-2 的线搜索函数 |ls_csrch| ，详见
% <ls_csrch.html 线搜索函数>
% ，或参考 Fortran 版本的官方代码 <https://ftp.mcs.anl.gov/pub/MINPACK-2/ MINPACK-2>。
%
% 此页面的源代码请见： <../download_code/newton/fminNewton.m fminNewton.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将