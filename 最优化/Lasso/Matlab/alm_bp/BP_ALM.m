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
%   2020.2.10 (Jiang Hu):
%     First version
%
%   2020.7.5 (Jiang Hu):
%     Parameter tuning
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 基追踪问题的增广拉格朗日函数法
% 考虑基追踪问题
%
% $$ \displaystyle \min_x\quad\|x\|_1,\quad \mathrm{s.t.}\quad Ax=b. $$
%
% 引入拉格朗日乘子 $\lambda$，增广拉格朗日函数可以写为
%
% $$ \displaystyle L_\sigma(x,\lambda)=\|x\|_1+\lambda^\top(Ax-b)
% +\frac{\sigma}{2}\|Ax-b\|_2^2, $$
% 其中 $\sigma$ 为罚因子。
%
% 增广拉格朗日函数法的迭代格式为
%
% $$
% \begin{array}{rl}
%   x^{k+1}=&\displaystyle\hspace{-0.5em}\arg\min\left\{\|x\|_1
%   +\frac{\sigma}{2}\left\|Ax-b+\frac{\lambda^k}{\sigma}\right\|_2^2\right\}, \\
%   \lambda^{k+1}=&\hspace{-0.5em}\lambda^k+\sigma (Ax^{k+1}-b).
% \end{array}
% $$
%
% 注意到关于 $x$ 的子问题没有显式解。我们利用近似点梯度法对其迭代求解。对应于外层第 $k$ 步迭代，内层迭代格式为
%
% $$ x^{s+1}=\mathrm{prox}_{\alpha_k\|\cdot\|_1}(x^s-\alpha_k \nabla\psi(x^s)),$$
%
% 其中
% $\psi(x)=\frac{\sigma}{2}\|Ax-b+\frac{\lambda^k}{\sigma}\|_2^2$，
% $\alpha_k$ 为步长。

%% 初始化和迭代准备
% 输入信息: 初始迭代点 $x^0$ 和数据 $A$, $b$，以及停机准则参数结构体 |opts| 。
%
% 输出信息: 迭代得到的解 $x$ 和包含算法求解中的相关迭代信息结构体 |out| 。
%
% * |out.feavec| ：每一步外层迭代的约束违反度 $\|Ax-b\|_2$
% * |out.inn_itr| ：总的内层迭代的步数（即近似点梯度法的迭代步数）
% * |out.tt| ：程序运行时间
% * |out.fval| ：迭代结束时的目标函数值
% * |out.itr| ：外层迭代步数
% * |out.itervec| ：迭代点列 $x$

function [x, out] = BP_ALM(x0, A, b, opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.itr| ：外层迭代的最大迭代步数
% * |opts.itr_inn| ：内层迭代的最大迭代步数
% * |opts.sigma| ：罚因子
% * |opts.tol0| ：初始的收敛精度
% * |opts.gamma| ：线搜索参数
% * |opts.verbose| ：表示输出的详细程度， |>1| 时每一步输出， |=1| 时每一次外层循环输出， |=0| 时不输出
if ~isfield(opts, 'itr'); opts.itr = 20; end
if ~isfield(opts, 'itr_inn'); opts.itr_inn = 2500; end
if ~isfield(opts, 'sigma'); opts.sigma = 1; end
if ~isfield(opts, 'tol0'); opts.tol0 = 1e-1; end
if ~isfield(opts, 'gamma'); opts.gamma = 1; end
if ~isfield(opts, 'verbose'); opts.verbose = 2; end
sigma = opts.sigma;
gamma = opts.gamma;

%%%
% 迭代准备。
k = 0;
tt = tic;
out = struct();

%%%
% 计算并记录初始时刻的约束违反度。
out.feavec = norm(A*x0 - b);
x = x0;
lambda = zeros(size(b));
out.itr_inn = 0;

%%%
% 记录迭代过程的优化变量 $x$ 的值。
itervec = zeros(length(x0),opts.itr);

%%%
% $\sigma A^\top A$ 的最大特征值，用于估计步长。
L = sigma*eigs(A'*A,1);

%% 迭代主循环
% 以 |opts.itr| 为最大迭代次数。
while k < opts.itr  
    %%%
    % 计算函数值和可微部分的梯度。
    % 函数值 $\|x\|_1+\frac{\sigma}{2}\|Ax-b+\frac{\lambda}{\sigma}\|^2_2$，
    % 可微部分的梯度 $g=\sigma A^\top(Ax-b+\frac{\lambda}{\sigma})$。
    Axb = A*x - b;
    c = Axb + lambda/sigma;
    g = sigma*(A'*c);
    tmp = .5*sigma*norm(c,2)^2;
    f = norm(x,1) + tmp;
    
    %%%
    % $x$-子问题的最优性条件违反度作为停机准则依据。
    % 令第 $k$ 步的停机精度阈值为 $ \tol_0 = 10^{-k}$，当最优性条件违反度小于此值时停止迭代。
    nrmG = norm(x - prox(x - g,1),2);
    tol_t = opts.tol0*10^(-k);
    t = 1/L;
    
    %% 子问题求解的近似点梯度法
    % 对于 $x$-子问题
    %
    % $$ \displaystyle x^{k+1}=\arg\min_x\left\{\|x\|_1+\frac{\sigma}{2}
    %    \left\|Ax-b+\frac{\lambda^k}{\sigma}\right\|_2^2 \right\} $$
    %
    % 使用近似点梯度法求解。 |k1| 记录内层迭代次数， |Cval| ， |Q| 为线搜索参数。
    Cval = tmp; Q = 1;
    k1 = 0;
    
    %%%
    % 内层循环，当最优性条件违反度小于阈值，或者超过最大内层迭代次数限制时，退出内层循环。
    while k1 < opts.itr_inn && nrmG > tol_t
        
        %%%
        % 记录上一步的结果。
        gp = g;
        xp = x;
        
        %%%
        % 进行一步近似点梯度法迭代，检验是否满足非精确线搜索条件。
        %
        % 事实上，对于内层迭代的第 $s$ 步，近似点梯度法的迭代格式 $x^{s+1}=\mathrm{prox}_{\alpha_s
        % \|\cdot\|_1}(x^{s}-\alpha_s \nabla \psi(x^s))$， 其中 $\psi(x) = \|Ax-b+\frac{\lambda^s}{\sigma}\|^2_2$， 根据定义即为
        %
        % 进一步计算，有
        %
        % $$
        % \begin{array}{rl}
        % x^{s+1}=&\displaystyle\hspace{-0.5em}\arg\min_u\left\{\|u\|_1+\frac{1}{2\alpha_k}\|u-x^s+\alpha_k\nabla \psi(x^s)\|_2^2 \right\} \\ 
        %        =&\displaystyle\hspace{-0.5em}\arg\min_u\left\{\|u\|_1+\psi(x^s)+\nabla
        %         \psi(x^s)^\top (u-x^s)+\frac{1}{2\alpha_s}\|u-x^s\|^2_2 \right\}.
        % \end{array}
        % $$
        %
        % 令 $\phi_s(x) =
        % \psi(x^s)+\nabla \psi(x^s)^\top (x-x^s)+\frac{1}{2\alpha_s}\|x-x^s\|^2_2,$
        % 以 $\phi_s(x)$的值的下降量为依据进行线搜索，即
        %
        % $$ \phi_s(x^s+\alpha d^s)\le C_s+\rho\alpha\nabla \phi_s(x^s)^\top d^s, $$
        %
        % 其中 $C_s$ 按照 (Zhang & Hager) 线搜索准则中的定义来计算。
        % 
        % |nls| 记录线搜索次数，直到满足下降量条件或进行 5 次步长衰减后退出线搜索循环。
        
        x = prox(xp - t*gp, t);
        nls = 1;
        
        while 1
            tmp = 0.5 *sigma*norm(A*x - b + lambda/sigma, 2)^2;
            if tmp <= Cval + g'*(x-xp) + .5*sigma/t*norm(x-xp,2)^2 || nls == 5
                break;
            end
            %%%
            % 如果没有达到线搜索标准，衰减步长，重新试探。
            t = 0.2*t;
            nls = nls + 1;
            x = prox(xp - t * g, t);
        end
        
        %%%
        % 退出线搜索后，更新各量的值。
        f = tmp + norm(x,1);
        nrmG = norm(x - xp,2)/t;
        Axb = A*x - b;
        c = Axb + lambda/sigma;
        g = sigma*(A' * c);
        
        %%%
        % 可微部分 BB 步长的计算，分别对应 $\displaystyle\frac{(s^k)^\top s^k}{(s^k)^\top y^k},$
        % $\displaystyle\frac{(s^k)^\top y^k}{(y^k)^\top y^k}$ 两个BB步长，
        % 其中 $s^k=x^{k+1}-x^k$, $y^k=g^{k+1}-g^{k}$。
        dx = x - xp;
        dg = g - gp;
        dxg = abs(dx'*dg);
        if dxg > 0
            if mod(k,2) == 0
                t = norm(dx,2)^2/dxg;
            else
                t = dxg/norm(dg,2)^2;
            end
        end
        
        %%%
        % 取步长为上述 BB 步长和 $1/L$中的较大者，并限制在 $10^{12}$ 范围内。
        t = min(max(t,1/L),1e12);
        
        %%%
        % 计算 (Zhang & Hager) 线搜索准则中的递推常数，其满足 $C_0=f(x^0),\ C_{k+1}=(\gamma
        % Q_kC_k+f(x^{k+1}))/Q_{k+1}$，序列 $Q_k$ 满足 $Q_0=1,\ Q_{k+1}=\gamma
        % Q_{k}+1$。
        Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + tmp)/Q;
        
        %%%
        % 当需要详细输出时，输出每一步的结果，利用 |k1| 记录内层迭代的迭代步数。
        k1 = k1 + 1;
        if opts.verbose > 1
            fprintf('itr_inn: %d\tfval: %e\t nrmG: %e\n', k1, f,nrmG);
        end
    end
    
    %%%
    % 每一次内层迭代结束输出当前的结果。
    if opts.verbose
        fprintf('itr_inn: %d\tfval: %e\t nrmG: %e\n', k1, f,nrmG);
    end
    
    %% 更新拉格朗日乘子
    % 在通过近似点梯度法对 $x$ 进行更新之后，对拉格朗日乘子 $\lambda$ 进行更新，迭代格式为
    % $\lambda^{k+1}=\lambda^k+\sigma(Ax^{k+1}-b)$。
    lambda = lambda + sigma*Axb;
    
    %%%
    % 迭代步数加 1，依次记录外层迭代后的约束违反度、每一步迭代的 $x$ 的值，更新内层迭代的总次数。
    k = k + 1;
    out.feavec = [out.feavec; norm(Axb)];
    itervec(:,k) = x;
    out.itr_inn = out.itr_inn + k1;
end

%%%
% 外层迭代结束，记录输出。
out.tt = toc(tt);
out.fval = f;
out.itr = k;
out.itervec = itervec;
end

%% 辅助函数
% 函数 $h(x)=\mu\|x\|_1$ 对应的邻近算子 $\mathrm{sign}(x)\max\{|x|-\mu,0\}$。
function y = prox(x,mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end

%% 参考页面
% 在页面 <demo_alm.html 实例：利用增广拉格朗日函数法解基追踪问题> 中，我们展示该算法的应用和数值表现。
%
% 此页面的源代码请见： <../download_code/alm_bp/BP_ALM.m BP_ALM.m>。

%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将
