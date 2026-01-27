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


%% RMSProp 算法
% 考虑优化问题：
%
% $$ \displaystyle \min_{x\in\mathcal{R}^n}f(x)=\frac{1}{N}\sum_{i=1}^Nf_i(x). $$
%
% RMSProp (root mean square propagation) 算法是对 AdaGrad 算法的改进。在
% AdaGrad 算法中，由于梯度分量的直接累加，步长随着迭代的进行而单调递减，
% 这可能导致后期步长过小。PMSProp 则在此基础上引入权重参数 $\rho$，其对应的分量累计变为加权和
%
% $$ M^{k+1}=\rho M^k+(1-\rho)g^{k+1}\odot g^{k+1}. $$
%
% 利用 $M^k$ 替代 AdaGrad 算法中的 $G^k$ 来对步长各分量进行修正。具体的迭代格式为：
%
% $$ \begin{array}{rl}
% x^{k+1}=&\hspace{-0.5em}\displaystyle x^k-\frac{\alpha}
% {\sqrt{M^k+\epsilon \mathbf{1}_n}}\odot g^k, \\
% M^{k+1}=&\hspace{-0.5em}\displaystyle \rho M^k+(1-\rho)g^{k+1}\odot g^{k+1}.
% \end{array} $$
%
%% 初始化和迭代准备
% 输入信息：迭代初始值 |x0| ，数据集大小 |N| ，样本梯度计算函数 |pgfun| ，目标函数值与梯度计算函数
% |fun| 以及提供算法参数的结构体 |opts| 。
% 
% 输出信息：迭代得到的解 $x$和包含迭代信息的结构体 |out| 。
%
% * |out.fvec| ：迭代过程中的目标函数值信息
% * |out.nrmG| ：迭代过程中的梯度范数信息
% * |out.epoch| ：迭代过程中的时期 (epoch)信息
function [x,out] = RMSProp(x0,N,pgfun,fun,opts)
%%%
% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大迭代次数
% * |opts.alpha| ：初始步长
% * |opts.thres| ：增强数值稳定性的小常量
% * |opts.rho| ：分量累积的权重
% * |outs.batchsize| ：随机算法的批量大小
% * |opts.verbose| ：不为 0 时输出每步迭代信息，否则不输出
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'alpha'); opts.alpha = 1e-2; end
if ~isfield(opts, 'thres'); opts.thres = 1e-7; end
if ~isfield(opts, 'rho'); opts.rho = 0.9; end
if ~isfield(opts, 'batchsize'); opts.batchsize = 10; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
%%%
% 以 |x0| 为迭代初始点。
% 计算初始点处的目标函数值和梯度，记初始时期 (epoch) 为 0。
x = x0;
out = struct();
[f,g] = fun(x);
out.fvec = f;
out.nrmG = norm(g,2);
out.epoch = 0;
%%%
% |gsum| 用来存储梯度分量的累计量。 |count| 用于计算时期。
gsum = zeros(size(x));
rho = opts.rho;
count = 1;
%% 迭代主循环
% RMSProp 的迭代循环，以 |opts.maxit| 为最大迭代次数。
for k = 1:opts.maxit
    %%%
    % 等概率地从 $\{1,2,\dots,N\}$ 中选取批量 $s_k$ 记录在 |idx| 之中，批量大小为
    % |opts.batchsize|. 计算对应的样本的梯度。
    idx = randi(N,opts.batchsize,1);
    g = pgfun(x,idx);
    %%%
    % RMSProp 改进的自适应步长方法，每一步的梯度分量积累为当前梯度分量和上一步分量积累的加权平均。
    %
    % $$ \displaystyle M^{k+1}=\rho M^k+(1-\rho)g^{k+1}\odot g^{k+1}. $$
    %
    if k == 1
        gsum = g.*g;
    else
        gsum = rho*gsum + (1-rho)*(g.*g);
    end
    %%%
    % 以梯度分量积累来确定逐分量步长，在下降更快的方向的步长减小，而下降更慢的方向以更大的步长进行更新。
    %
    % $$ \displaystyle x^{k+1}=x^k-\frac{\alpha}{\sqrt{M^k+\epsilon \mathbf{1}_n}}\odot g^k. $$
    %
    x = x - opts.alpha./sqrt(gsum + opts.thres).*g;
    %%%
    % 每当参与迭代的总样本次数超过数据集的总样本时，记为一个时期 (epoch)。
    % 每一个时期，记录当前的目标函数值和梯度范数，并令时期计数加一。
    if k*opts.batchsize/N >= count
        [f,g] = fun(x);
        out.fvec = [out.fvec; f];
        out.nrmG = [out.nrmG; norm(g,2)];
        out.epoch = [out.epoch; k*opts.batchsize/N];
        count = count + 1;
    end
end
end
%% 参考页面
% 在页面 <demo_lr_sg.html 实例：利用随机算法求解逻辑回归问题> 
% 中，我们展示该算法的一个应用，并且与其它随机算法进行比较。
%
% 其它随机算法参见： <sgd.html 随机梯度下降法>、
% <Adagrad.html AdaGrad>、 <AdaDelta.html AdaDelta>、 <Adam.html Adam>。
%
% 此页面的源代码请见： <../download_code/stograd/RMSProp.m RMSProp.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将
