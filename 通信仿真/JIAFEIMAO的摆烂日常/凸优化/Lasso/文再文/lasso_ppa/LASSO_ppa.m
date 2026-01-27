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
%   2020.7.15 (Jiang Hu):
%     Parameter tuning
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LASSO 问题的近似点算法
% 近似点算法（PPA）对于 LASSO 问题，考虑其等价形式
%
% $$ \min_{x,y} f(x,y)=\mu\|x\|_1+\frac{1}{2}\|y\|_2^2,
% \quad \mathrm{s.t.}\quad Ax-y-b=0.$$
%
% 近似点算法的一个迭代步为
%
% $$ (x^{k+1},y^{k+1})\approx\arg\min_{(x,y)\in\mathcal{D}}
% \left\{f(x,y)+\frac{1}{2t_k}(\|x-x^k\|_2^2+\|y-y^k\|_2^2)\right\}, $$
%
% 其中 $\mathcal{D}=\{(x,y)\big|Ax-y=b\}$。对于子问题考虑其对偶问题，引入拉格朗日乘子 $z$，并令
%
% $$ \Phi_k(z)=\inf_x\left\{\mu\|x\|_1+z^\top Ax+\frac{1}{2t_k}\|x-x^k\|_2^2
% \right\} +\inf_y\left\{\frac{1}{2}\|y\|_2^2-z^\top y+\frac{1}{2t_k}
% \|y-y^k\|_2^2\right\}-b^\top z, $$
%
% 则其迭代格式满足
%
% $$ z^{k+1}=\arg\max_z \Phi_k(z).$$
%
% 以 $z^{k+1}$ 为逼近解， $\inf_x$ 和 $\inf_y$ 的子问题是半光滑的，
% 因此利用半光滑牛顿法加速的梯度法进行求解。
% 再结合最优性条件，有
%
% $$ \left\{\begin{array}{l}x^{k+1}=\mathrm{prox}_{\mu t_k\|\cdot\|_1}
% (x^k-t_kA^\top z^{k+1})\\y^{k+1}=\frac{1}{t_k+1}(y^k+t_kz^{k+1})\end{array}\right. $$
%
%% 初始化和迭代准备
% PPA不使用连续化策略，直接对原始的 LASSO 问题进行求解。
%
% 输入信息： $A$, $b$,
% $\mu$ ，迭代初始值 $x^0$ 以及提供各参数的结构体 |opts| 。
%
% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。
%
% * |out.fvec| ：每一步迭代的 LASSO 目标函数值
% * |out.flag| ：标记是否达到收敛
% * |out.fval| ：迭代终止时的 LASSO 目标函数值
% * |out.tt| ：运行时间
% * |out.itr| ：外层迭代次数
function [x, out] = LASSO_ppa(x0, A, b, mu, opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：针对函数值的停机准则，当相邻两次迭代函数值之差小于该值时认为该条件满足
% * |opts.gtol| ：针对梯度的停机准则，当当前步梯度范数小于该值时认为该条件满足
% * |opts.verbose| ： |>1| 时输出每一步信息， |0-1| 时输出外层迭代信息， |=0| 时不输出
% * |opts.t0| ：初始步长
if ~isfield(opts, 'maxit'); opts.maxit = 500; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
if ~isfield(opts, 't0'); opts.t0 = 1e3; end

out = struct();
%%%
% optsz 为 z 的子问题提供参数。
optsz = struct();
optsz.verbose = (opts.verbose > 1);

%%%
% 迭代初始化。
%
% 以 x^0 为迭代初始点，在初始 x^0 处计算相关变量的值。
x = x0;
fp = inf;
tt = tic;
r = A*x - b;
f = .5*norm(r,2)^2 + mu*norm(x,1);
out.fvec = f;
%%%
% t 为步长，初始值设定为 opts.t0。
t = opts.t0;

%%%
% 变量 y=Ax-b 和拉格朗日乘子 z。
z = zeros(size(b));
y = A*x - b;
toldiff = 1;

%% 迭代主循环
% 以 |maxit| 为最大迭代步数。每次迭代开始，记录上一次迭代的目标函数值。
for k = 1:opts.maxit
    fp = f;
    %%%
    % |toldiff| 为 $\|x^{k+1}-x^{k}\|_2^2+\|y^{k+1}-y^{k}\|_2^2$，这里为内层的
    % $z$ 的子问题迭代确定停机准则：
    % $\|\nabla\Phi_k(z^{k+1})\|_2\le\sqrt{\alpha_k/t_k}\sigma_k\|
    % (x^{k+1},y^{k+1})-(x^k,y^k)\|_2$，并令其中的 $\delta_k=\frac{8}{k^2}$，
    % $\alpha_k=\frac{t_k}{t_k+1}$。
    optsz.gtol = 8/k^2*min(1, toldiff);
    %%%
    % 内层循环的梯度法解拉格朗日乘子 $z$ 的子问题。然后对 $x$, $y$ 更新。
    %
    %
    % $$ \begin{array}{rl}
    %   x^{k+1}=&\hspace{-0.5em}\mathrm{prox}_{\mu t_k\|\cdot\|_1}
    %   (x^k-t_kA^\top z^{k+1}),\\
    %   y^{k+1}=&\hspace{-0.5em}\frac{1}{t_k+1}(y^k+t_kz^{k+1}).
    % \end{array} $$
    %
    [z,~] = zgbb(z,A,b,mu,t, x,y, optsz);
    xp = x;
    x = prox(x - t*A'*z, mu*t);
    yp = y;
    y = 1/(t+1)*(y + t*z);
    %%%
    % 一步 PPA 的外层迭代结束，计算目标函数值，并以一步近似点梯度法的 $x$ 估计梯度。
    Axb = A*x - b;
    f = .5*norm(Axb,2)^2 + mu*norm(x,1);
    nrmG = norm(x - prox(x - A'*Axb, mu));
    toldiff = norm(x-xp,2) + norm(y- yp,2);
    
    %%%
    % 记录一步迭代的目标函数值。
    out.fvec = [out.fvec; f];
    
    %%%
    % 当 opts.verbose 不为 0 时输出详细的迭代信息。
    if opts.verbose
        fprintf('itr: %d\t t: %e\t fval: %e \t nrmG: %.1e \t feasi: %.1e \n', ...
        k, t, f, nrmG, norm(A*x -b - y,2));
    end
    %%%
    % 外层迭代的停机准则，当函数值的变化或梯度（一步近似点梯度法估计）范数小于阈值时，
    % 认为收敛，退出循环。
    if abs(f-fp) < opts.ftol || nrmG < opts.gtol
        break;
    end
end

%%%
% 当收敛时，记录输出信息。以 |out.flag| 记录迭代结束原因， |out.flag=1| 表示满足梯度条件，
% out.flag=0 表示满足函数值条件。
if abs(f - fp) < opts.ftol
    out.flag = 0;
else
    out.flag = 1;
end
%%%
% 记录输出信息。
out.fval = f;
out.itr = k;
out.tt = toc(tt);
end
%% 拉格朗日乘子 $z$ 子问题
%%%
% 从输入的结构体中读取参数或采取默认参数。
%
% * |optsz.maxit| ：最大迭代次数
% * |optsz.gtol| ：子问题的停机准则，当梯度范数小于该值时认为满足；
% 注意到该值随着外层迭代而变化，具体的计算方法见上文
% * |optsz.verbose| ：不为 0 时输出每步迭代信息，否则不输出
% * |optsz.tau0| ：初始步长
function [z,out] = zgbb(z0,A,b,mu,t,x,y, optsz)
if ~isfield(optsz, 'maxit'); optsz.maxit = 500; end
if ~isfield(optsz, 'gtol'); optsz.gtol = 1e-6; end
if ~isfield(optsz, 'verbose'); optsz.verbose = 0; end
if ~isfield(optsz, 'tau0'); optsz.tau0 = 1e-2; end
%%%
% 初始化变量。
out = struct();
[m,n] = size(A);
Im = eye(m);
H = eye(m);
z = z0;
r = x - t*(A'*z);
w = prox(r,mu*t);
%%%
% 初始步长设定为 optsz.tau0。
tau = optsz.tau0;
%%%
% 将子问题中的 $y$ 的部分解析地代入，得到其迭代的目标函数(去掉常数项)
%
% $$ -\Phi_k(z)=-\mu\Gamma_{\mu t_k}(x^k-t_kA^\top z)+\frac{1}{2t_k}
% (\|x^k-t_kA^\top z\|_2^2)+\frac{t_k}{2(t_k+1)}\|z\|_2^2+\frac{1}{t_k+1}z^\top
% y^k-\frac{1}{2(t_k+1)}\|y^k\|_2^2+b^\top z, $$
%
% 其中 $\Gamma_{\mu t_k}(u)=\inf_x\left\{\|x\|_1+\frac{1}{2\mu
% t_k}\|x-u\|_2^2\right\}$。
%
% 令 $w=\mathrm{prox}_{\mu t_k\|\cdot\|_1}(x^k-t_kA^\top z)$, $\Gamma_{\mu
% t_k}(x^k-t_kA^\top z)=\|w\|_1+\frac{1}{2\mu^2t_k}\|w-(x^k-t_kA^\top
% z)\|_2^2$。
f = - mu*(norm(w,1) + 1/(2*mu*t)*norm(w-r,2)^2) + 1/(2*t)*norm(r,2)^2 ...
+ t/(2*(t+1))*(norm(z,2)^2) + 1/(t+1)*(y'*z) + b'*z;
%%%
% 初始化线搜索参数。
Q = 1; Cval = f; gamma = 0.85; rhols = 1e-6;
%%%
% 目标函数为连续可微函数，且其梯度为 $-Aw+\frac{t_k}{t_k+1}z+\frac{1}{t_k+1}y+b$。
g = - A*w + t/(t+1)*z + 1/(t+1)*y + b;
nrmg = norm(g,2);
%%%
% 对 $z$ 的梯度法迭代主循环，最大迭代步 |maxit|。
maxit = optsz.maxit;
for k = 1:maxit
    zp = z;
    gp = g;
    %%%
    % 进行线搜索。 |nls| 记录线搜索次数， $\eta=0.2$ 表示每次不满足线搜索时步长减小的比例。
    % 每次循环以当前步长进行一步试探，下降方向 $d^k=-H g^k$，求出在新的 $z$ 处的变量和目标函数值。
    % 当 $H=I$ 时，则为一般的带线搜索的 BB 步长梯度法，从 10 步后， $H$
    % 为广义海瑟矩阵的逆，步长设定为 $\tau_k=1$，对应半光滑牛顿法。
    nls = 1; deriv = rhols*nrmg^2; eta = 0.2;
    
    while 1
        z = zp - tau*(H*g);
        r = x - t*(A'*z);
        w = prox(r, mu*t);
        f = - mu*(norm(w,1) + 1/(2*mu*t)*norm(w-r,2)^2) + 1/(2*t)*norm(r,2)^2 ...
        + t/(2*(t+1))*(norm(z,2)^2) + 1/(t+1)*(y'*z) + b'*z;
        %%%
        % 满足线搜索准则 (Zhang & Hager) $\displaystyle f(x^k+\tau d^k)\le C_k
        % +\rho\tau (g^k)^\top d^k$ 或进行超过 10 次步长衰减后退出线搜索，否则以 $\eta$
        % 的比例对步长进行衰减。
        if f < Cval - tau*deriv || nls == 10
            break;
        end
        tau = eta*tau;
        nls = nls + 1;
    end
    
    %%%
    % 更新梯度。
    g = - A*w + t/(t+1)*z + 1/(t+1)*y + b;
    nrmg = norm(g,2);
    
    %%%
    % 当 |optsz.verbose| 不为 0 时输出详细的迭代信息。
    if optsz.verbose
        fprintf('iter: %d \t f: %.4e \t nrmG: %.1e \t nls: %d  \n', k, f, nrmg, nls);
    end
    
    %%%
    % 停机准则（见上文），当满足停机准则时退出循环。
    gtol = sqrt(1/(t+1))*optsz.gtol;
    if nrmg < gtol
        break;
    end
    %%%
    % BB 步长的计算，令 $s^k=z^{k+1}-z^k$, $y^k=g^{k+1}-g^k$，则两种 BB 步长
    % $\displaystyle \tau_{BB1}=\frac{(s^k)^\top s^k}{(s^k)^\top y^k}$,
    % $\displaystyle \tau_{BB2}=\frac{(s^k)^\top y^k}{(y^k)^\top y^k}$。
    % 这里我们交替使用两种 BB 步长并限定在 $[10^{-12},10^{12}]$ 中。
    dz = z - zp;
    dg = g - gp;
    dzg = abs(dz'*dg);
    if dzg > 0
        if mod(k,2) == 0
            tau = norm(dz,2)^2 /dzg;
        else
            tau = dzg/(norm(dg,2)^2);
        end
    end
    tau = min(max(1e-12,tau),1e12);
    
    %%%
    % 线搜索参数的更新。
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + f)/Q;
    %%%
    % 从 10 步之后，开始使用半光滑牛顿法。
    % 由优化的目标函数的形式
    %
    % $$ \displaystyle -\Phi_k(z)=-\mu\Gamma_{\mu t_k}(x^k-t_kA^\top z)
    % +\frac{1}{2t_k}(\|x^k-t_kA^\top z\|_2^2)+\frac{t_k}{2(t_k+1)}\|z\|_2^2
    % +\frac{1}{t_k+1}z^\top y^k-\frac{1}{2(t_k+1)}\|y^k\|_2^2+b^\top z, $$
    %
    % 得到其广义海瑟矩阵
    %
    % $$ \displaystyle -\nabla^2\Phi_k(z)=\frac{t_k}{t_k+1}I+t_k\bar{A}\bar{A}^\top, $$
    %
    % 其中 $\bar{A}$ 为满足 $x^k-t_kA^\top z>\mu t_k$ 的下标对应的 $A$ 的列。再由
    % Sherman-Morrison-Woodbury 公式可得
    %
    % $$ \displaystyle H=\frac{t_k+1}{t_k}\left(I-\bar{A}(\frac{1}{t_k+1}I
    % +\bar{A}^\top\bar{A})^{-1}\bar{A}^\top\right) $$
    %
    % 为广义海瑟矩阵的逆。在利用半光滑牛顿法的情况下，采用 $\tau=1$ 的固定步长和一般的线搜索准则
    % （即 |Cval| 直接取当前点函数值）。
    if k > 10
        ind = find(abs(r) > mu*t);
        Aind = A(:,ind);
        tmp = 1/(t+1)*eye(length(ind)) + Aind'*Aind;
        H = (t+1)/t*(Im - Aind*(tmp\Aind'));
        tau = 1; Cval = f;
    end
end
%%%
% 当从迭代中退出时，返回这一内重循环的迭代步。
out.iter = k;
end
%% 辅助函数
%%%
% 函数 $h(x)=\mu\|x\|_1$ 对应的邻近算子 $\mathrm{sign}(x)\max\{|x|-\mu,0\}$。
function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end
%% 参考页面
% 在页面 <demo_ppa.html 实例：近似点法解 LASSO 问题> 中我们构建一个 LASSO
% 问题并展示该算法在其中的应用。
%
% 此页面的源代码请见： <../download_code/lasso_ppa/LASSO_ppa.m LASSO_ppa.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将