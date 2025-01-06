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


%% L-BFGS 解优化问题
% 针对无约束优化问题
%
% $$ \min_x  f(x),$$
%
% L-BFGS 在拟牛顿法 BFGS 迭代格式的基础上进行修改，
% 用以解决大规模问题的存储和计算困难。对于拟牛顿法中的迭代方向 $d^k=H^k\nabla f(x^k)$
% 考虑利用递归展开的方式进行求解。
%
% 首先，对于 BFGS 迭代格式， $H^{k+1}=(V^k)^\top H^kV^k+\rho_ks^k(s^k)^\top$，其中
% $\displaystyle\rho_k = \frac{1}{(y^k)^\top s^k},$ $V^k=I-\rho_k
% y^k(s^k)^\top$。
% 将其递归地展开得到
%
% $$ \displaystyle\begin{array}{rl} -H^k\nabla f(x^k)= & -(V^{k-m}\cdots
% V^{k-1})^\top H^{k-m}(V^{k-m}\cdots V^{k-1})\nabla f(x^k)\\
% & -\rho_{k-m}(V^{k-m+1}\cdots V^{k-1})^\top s^{k-m}(s^{k-m})^\top
% (V^{k-m+1}\cdots V^{k-1})\nabla f(x^k)\\
% & -\rho_{k-m+1}(V^{k-m+2}\cdots V^{k-1})^\top s^{k-m+1}
% (s^{k-m+1})^\top(V^{k-m+2}\cdots V^{k-1})\nabla f(x^k)\\
% &-\cdots\\ &-\rho_{k-1}s^{k-1}(s^{k-1^\top})\nabla f(x^k).\end{array} $$
%
% 我们只需对其中的 $H^{k-m}$ 进行某种估计，即可在展开深度为 $m$ 的情况下对
% $d^k=-H^k\nabla f(x^k)$ 进行近似求解。当用数量矩阵来近似时，即
% $\hat{H}^{k-m}=\gamma_k I$，其中 $\displaystyle\gamma_k=\frac{(s^{k-1})^\top
% y^{k-1}}{(y^{k-1})^\top y^{k-1}}$ 对应 BB 方法的第二个步长。
%
% 第一个循环：初始化 $q^{k}=-\nabla f(x^k)$，迭代 $\alpha_i=\rho_i(s^i)^\top q^{i+1}$，
% $q^{i}=q^{i+1}-\alpha_i y^i$。其中 $i$ 从 $k-1$ 减小到 $k-m$。
% 不难证明 $q^{k-m}$ 递归地求出了 $-V^{k-m}\cdots V^{k-1}\nabla f(x^k)$ 并同时求出了
% $\alpha_{i}=-\rho_{i}(s^{i})^\top (V^{i+1}\cdots V^{k-1})\nabla f(x^k)$。
%
% 经过第一个循环，我们可以将上述展开的表达式重写为
%
% $$ \displaystyle \begin{array}{rl} -H^k\nabla f(x^k)=
% & -(V^{k-m}\cdots V^{k-1})H^{k-m}q^{k-m}\\
% & -(V^{k-m+1}\cdots V^{k-1})^\top s^{k-m}\alpha_{k-m}\\
% & -(V^{k-m+2}\cdots V^{k-1})^\top s^{k-m+1}\alpha_{k-m+1}\\ &-\cdots\\
% &-s^{k-1}\alpha_{k-1}.\end{array} $$
%
% 再引入第二个循环对这一求和式进行计算。
%
% 初始化 $r^{k-m}=\hat{H}^{k-m}q^{k-m}$，迭代
%
% $$ \begin{array}{rl}
% \beta_{i}=&\hspace{-0.5em}\rho_i(y^i)^\top r^i,\\
% r^{i+1}=&\hspace{-0.5em}r^i+(\alpha_i-\beta_i)s^i.
% \end{array} $$
%
% 可以验证每一次循环 $r$ 都相当于对求和式提取公因式后的从前到后进行一步累加，最终得到 $r^k$ 即为所需的
% $-H^k\nabla f(x^k)$。
%
% 利用上面双循环算法对下降方向进行求解的方法即为 L-BFGS 方法，这一方法只需记录 $m$ 步的信息
% $\{(s^i,y^i)\}_{i=k-m}^{k}$，在每一步更新时，将新得到的
% $(s^{k+1},y^{k+1})$ 覆盖 $(s^{k-m},y^{k-m})$，因此需要的空间大大减小。
%
% 该函数完成上述的 L-BFGS 算法，利用双循环算法计算下降方向，并利用线搜索确定步长。
%% 初始化和迭代准备
%%%
% 函数输入： |x| 为迭代的初始点， |fun| 提供函数值和梯度， |opts| 为提供算法参数的结构体。
%
% 函数输出： |x| 为迭代得到的解， |f| 和 |g| 为该点处的函数值和梯度， |Out| 
% 为记录迭代信息的结构体。
%
% * |Out.f| ：迭代过程的函数值信息
% * |Out.nrmG| ：迭代过程的梯度范数信息
% * |Out.xitr| ：迭代过程的优化变量 $x$（仅在 |storeitr| 不为 0 时存在）
% * |Out.nfe| ：调用目标函数的次数
% * |Out.nge| ：调用梯度的次数
% * |Out.nrmg| ：迭代终止时的梯度范数
% * |Out.iter| ：迭代步数
% * |Out.msg| ：标记是否达到收敛
function [x, f, g, Out]= fminLBFGS_Loop(x, fun, opts, varargin)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.xtol| ：主循环针对优化变量的停机判断依据
% * |opts.gtol| ：主循环针对梯度范数的停机判断依据
% * |opts.ftol| ：主循环针对函数值的停机判断依据
% * |opts.rho1| ：线搜索标准 $\rho_1$
% * |opts.rho2| ：线搜索标准 $\rho_2$
% * |opts.m| ：L-BFGS 的内存对存储数
% * |opts.maxit| ：主循环的最大迭代次数
% * |opts.storeitr| ：标记是否记录每一步迭代的 $x$
% * |opts.record| ：标记是否需要迭代信息的输出
% * |opts.itPrint| ：每隔几步输出一次迭代信息
if ~isfield(opts, 'xtol');      opts.xtol = 1e-6; end
if ~isfield(opts, 'gtol');      opts.gtol = 1e-6; end
if ~isfield(opts, 'ftol');      opts.ftol = 1e-16; end
if ~isfield(opts, 'rho1');      opts.rho1  = 1e-4; end
if ~isfield(opts, 'rho2');      opts.rho2  = 0.9; end
if ~isfield(opts, 'm');         opts.m  = 5; end
if ~isfield(opts, 'maxit');     opts.maxit  = 1000; end
if ~isfield(opts, 'storeitr');  opts.storeitr = 0; end
if ~isfield(opts, 'record');    opts.record = 0; end
if ~isfield(opts,'itPrint');    opts.itPrint = 1;   end
%%%
% 参数复制。
xtol = opts.xtol;
ftol = opts.ftol;
gtol = opts.gtol;
maxit = opts.maxit;
storeitr = opts.storeitr;
parsls.ftol = opts.rho1;
parsls.gtol = opts.rho2;
m = opts.m;
record = opts.record;
itPrint = opts.itPrint;

%%%
% 初始化和迭代准备，计算初始点处的信息。初始化迭代信息。
[f,  g] = feval(fun, x , varargin{:});
nrmx = norm(x);
Out.f = f;  Out.nfe = 1; Out.nrmG = [];

%%%
% 在 |storeitr| 不为 0 时，记录每一步迭代的 $x$。
if storeitr
    Out.xitr = x;
end
%%%
% |SK| , |YK| 用于存储 L-BFGS 算法中最近的 $m$ 步的 $s$（ $x$ 的变化量）和 $y$
% （梯度 $g$ 的变化量）。
n = length(x);
SK = zeros(n,m);
YK = zeros(n,m);
istore = 0; pos = 0;  status = 0;  perm = [];
%%%
% 为打印每一步的迭代信息设定格式。
if record == 1
    if ispc; str1 = '  %10s'; str2 = '  %6s';
    else     str1 = '  %10s'; str2 = '  %6s'; end
    stra = ['%5s',str2,str2,str1, str2, str2,'\n'];
    str_head = sprintf(stra, ...
        'iter', 'stp', 'obj', 'diffx', 'nrmG', 'task');
    str_num = ['%4d  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %2d\n'];
end

%% 迭代主循环
% 迭代最大步数 |maxit| 。当未达到收敛条件时，记录为超过最大迭代步数退出。
Out.msg = 'MaxIter';
for iter = 1:maxit
    %%%
    % 记录上一步迭代的结果。
    xp = x;   nrmxp = nrmx;
    fp = f;   gp = g;
    %%%
    % L-BFGS 双循环方法寻找下降方向。在第一次迭代时采用负梯度方向，之后便使用 L-BFGS 方法来
    % 估计 $d=-Hg$。
    if istore == 0
        d = -g;
    else
        d = LBFGS_Hg_Loop(-g);
    end
    %%%
    % 沿 L-BFGS 方法得到的下降方向做线搜索。调用函数 |ls_csrch| 进行线搜索，其参考了
    % MINPACK-2 中的线搜索函数。
    %
    % 首先初始化线搜索标记 |workls.task| 为 1， |deriv| 为目标函数沿当前下降方向的方向导数。
    % 通过线搜索寻找合适的步长 $\alpha_k$，使得以下条件满足：
    %
    % $$ \begin{array}{rl}
    % f(x^k+\alpha_k d^k)&\hspace{-0.5em}\le
    % f(x^k)+\rho_1\cdot\alpha_kg(x^k), \\
    % |g(x^k+\alpha_kd^k)|&\hspace{-0.5em}\le \rho_2\cdot |g(x^k)|.
    % \end{array} $$
    %
    % |ls_csrch| 每次调用只执行线搜索的一步，并用 |workls.task| 指示下一步应当执行的操作。
    % 此处 |workls.task==2| 意味着需要重新计算当前点函数值和梯度等。具体的步长寻找过程比较复杂，可以参考相应文件。
    %
    % 直到满足线搜索条件时，退出线搜索循环，得到更新之后的 $x$。
    workls.task =1;
    deriv = d'*g;
    normd = norm(d);
    
    stp = 1;
    while 1
        [stp, f, deriv, parsls, workls] = ....
            ls_csrch(stp, f, deriv , parsls , workls);
        
        if (workls.task == 2)
            x = xp + stp*d;
            [f,  g] = feval(fun, x, varargin{:});
            Out.nfe = Out.nfe + 1;
            deriv = g'*d;
        else
            break
        end
    end
    
    %%%
    % 对于线搜索得到的步长 $\alpha_k$，令 $s^k=x^{k+1}-x^k=\alpha_kd^k$，
    % 则 $\|s^k\|_2=\alpha_k\|d^k\|_2$。计算 $\|s^k\|_2/\max(1,\|x^k\|_2)$
    % 并将其作为判断收敛的标准。
    nrms = stp*normd;
    diffX = nrms/max(nrmxp,1);
    %%%
    % 更新 $\|x^{k+1}\|_2$, $\|g^{k+1}\|_2$，记录一步迭代信息。
    nrmG =  norm(g);
    Out.nrmg =  nrmG;
    Out.f = [Out.f; f];
    Out.nrmG = [Out.nrmG; nrmG];
    if storeitr
        Out.xitr = [Out.xitr, x];
    end
    nrmx = norm(x);
    %%%
    % 停机准则， |diffX| 表示相邻迭代步 $x$ 的相对变化， |nrmG| 表示当前 $x$ 处的梯度范数， $\displaystyle\frac{|f(x^{k+1})-f(x^k)|}{1+|f(x^k)|}$
    % 用以表示函数值的相对变化。当前两者均小于阈值，或者第三者小于阈值时，认为达到停机标准，退出当前循环。
    cstop = ((diffX < xtol) && (nrmG < gtol) )|| (abs(fp-f)/(abs(fp)+1)) < ftol;
    %%%
    % 当需要详细输出时，在（1）开始迭代时（2）达到收敛时（3）达到最大迭代次数或退出迭代时（4）每若干步，打印详细结果。
    if (record == 1) && (cstop || iter == 1 || iter==maxit || mod(iter,itPrint)==0)
        if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit && ~cstop
            fprintf('\n%s', str_head);
        end
        fprintf(str_num, ...
            iter, stp, f, diffX, nrmG, workls.task);
    end
    %%%
    % 当达到收敛条件时，停止迭代，记为达到收敛。
    if cstop
        Out.msg = 'Converge';
        break;
    end
    %%%
    % 计算 $s^k=x^{k+1}-x^{k}$, $y^k=g^{k+1}-g^{k}$。
    % 当得到的 $\|y^k\|$ 不小于阈值时，保存当前的 $s^k, y^k$，否则略去。利用 |pos|
    % 记录当前存储位置，然后覆盖该位置上原来的信息。
    ygk = g-gp;		s = x-xp;
    if ygk'*ygk>1e-20
        istore = istore + 1;
        pos = mod(istore, m); if pos == 0; pos = m; end
        YK(:,pos) = ygk;  SK(:,pos) = s;   rho(pos) = 1/(ygk'*s);
        %%%
        % 用于提供给 L-BFGS 双循环递归算法，以指明双循环的循环次数。当已有的记录超过 $m$ 时，
        % 则循环 $m$ 次。否则，循环次数等于当前的记录个数。 |perm| 按照顺序记录存储位置。
        if istore <= m; status = istore; perm = [perm, pos];
        else status = m; perm = [perm(2:m), perm(1)]; end
    end
end
%%%
% 当从上述循环中退出时，记录输出。
Out.iter = iter;
Out.nge = Out.nfe;
%% L-BFGS 双循环递归算法
% 利用双循环递归算法，返回下一步的搜索方向即 $-Hg$。
% 初始化 $q$ 为初始方向，在 L-BFGS 主算法中，这一方向为负梯度方向。
    function y = LBFGS_Hg_Loop(dv)
        q = dv;   alpha = zeros(status,1);
        %%%
        % 第一个循环， |status| 步迭代。（ |status| 的计算见上，当迭代步足够大时为 $m$ ）
        % |perm| 按照顺序记录了存储位置。从中提取出位置 $k$ 的格式为：
        % $\alpha_i=\rho_i(s^i)^\top q^{i+1}$ ,
        % $q^{i}=q^{i+1}-\alpha_i y^i$。其中 $i$ 从 $k-1$ 减小到 $k-m$。
        
        for di = status:-1:1
            k = perm(di);
            alpha(di) = (q'*SK(:,k)) * rho(k);
            q = q - alpha(di)*YK(:,k);
        end
        %%%
        % $r^{k-m}=\hat{H}^{k-m}q^{k-m}.$
        y = q/(rho(pos)* (ygk'*ygk));
        %%%
        % 第二个循环，迭代格式 $\beta_{i}=\rho_i(y^i)^\top r^i$,
        % $r^{i+1}=r^i+(\alpha_i-\beta_i)s^i$。代码中的 |y| 对应于迭代格式中的 $r$，当两次循环结束时，以返回的 |y| 的值作为下降方向。
        for di = 1:status
            k = perm(di);
            beta = rho(k)* (y'* YK(:,k));
            y = y + SK(:,k)*(alpha(di)-beta);
        end
    end

end
%% 参考页面
% 关于 L-BFGS 算法的应用，参考页面 <demo_lr_lbfgs.html 实例：L-BFGS 解逻辑回归问题>
% 以及 <demo_bp_lbfgs.html 实例：L-BFGS 解基追踪问题>。
%
% 我们在其中利用了翻译为 MATLAB 代码的 MINPACK-2 的线搜索函数 |ls_csrch| ，
% 代码详见 <ls_csrch.html 线搜索函数> 和被其调用的
% <ls_dcstep.html 线搜索辅助函数> ，或参考 Fortran 版本的官方代码
% <https://ftp.mcs.anl.gov/pub/MINPACK-2/ MINPACK-2>。
%
% 此页面的源代码请见：
% <../download_code/lbfgs/fminLBFGS_Loop.m fminLBFGS_Loop.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将