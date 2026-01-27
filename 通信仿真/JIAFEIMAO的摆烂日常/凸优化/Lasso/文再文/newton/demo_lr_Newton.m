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


%% 实例：应用牛顿-共轭梯度法解逻辑回归问题
% 考虑逻辑回归问题
%
% $$ \displaystyle\min_x \frac{1}{m}\sum_{i=1}^m
% \log(1+ \exp(-b_ia_i^Tx)) + \mu\|x\|_2^2, $$
%
% 其中 $(a_i,b_i)_{i=1}^m$ 为已知的待分类的数据集。这里利用非精确牛顿法（牛顿-共轭梯度法）求解该优化问题。
%
% 该优化问题目标函数的梯度和海瑟矩阵：
%
% $$ \begin{array}{rl}
% \displaystyle\nabla f(x) &\displaystyle\hspace{-0.5em}= \frac{1}{m}\sum_{i=1}^m
% \frac{1}{1+\exp(-b_ia_i^Tx)} \cdot \exp(-b_ia_i^Tx) \cdot (-b_ia_i) +
% 2\mu x, \\
% \displaystyle H(x)&\displaystyle\hspace{-0.5em}= \frac{1}{m}\sum_{i=1}^m \frac{\exp(-b_ia_i^\top x)}
% {(1+\exp(-b_ia_i^\top x))^2}a_ia_i^\top + 2\mu I. 
% \end{array}$$
%
%% 逻辑回归问题
% 在不同的数据集上进行实验。导入 LIBSVM 数据集 a9a 上的实验， |libsvmread|
% 为另外运行的读入程序。
dataset = 'a9a.test';
[b,A] = libsvmread(dataset);
[m,n] = size(A);
mu = 1e-2/m;
%%%
% 设定参数。
opts = struct();
opts.xtol = 1e-8;
opts.gtol = 1e-6;
opts.ftol = 1e-16;
opts.verbose = 0;
fun = @(x) lr_loss(x,mu,A,b);
x0 = zeros(n,1);
hess = @(x,u) lr_hess(x,u,A,b,mu);
%%%
% 调用牛顿法求解。
[x1,out1] = fminNewton(x0,fun,hess,opts);

%%%
% 在 CINA 数据集上的实验。
dataset = 'CINA.test';
[b,A] = libsvmread(dataset);
[m,n] = size(A);
fun = @(x) lr_loss(x,mu,A,b);
x0 = zeros(n,1);
hess = @(x,u) lr_hess(x,u,A,b,mu);
[x2,out2] = fminNewton(x0,fun,hess,opts);

%%%
% 在 ijcnn1 数据集上的实验。
dataset = 'ijcnn1.test';
[b,A] = libsvmread(dataset);
Atran = A';
[m,n] = size(A);
mu = 1e-2/m;
fun = @(x) lr_loss(x,mu,A,b);
x0 = zeros(n,1);
hess = @(x,u) lr_hess(x,u,A,b,mu);
[x3,out3] = fminNewton(x0,fun,hess,opts);
%% 结果可视化
% 将目标函数梯度范数随着迭代步的变化可视化。
fig = figure;
semilogy(0:out1.iter, out1.nrmG, '-o', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:out2.iter, out2.nrmG, '-.*', 'Color',[0.99 0.1 0.2], 'LineWidth',1.8);
hold on
semilogy(0:out3.iter, out3.nrmG, '--d', 'Color',[0.99 0.1 0.99], 'LineWidth',1.5);
legend('a9a','CINA','ijcnn1');
ylabel('$\|\nabla \ell_(x^k)\|_2$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','lr_newton.eps');

%% 辅助函数
% 逻辑回归的损失函数，作为优化问题的目标函数。
% |nargout| 表示当前函数在被调用时，需要的输出的个数，当输出个数大于1时计算梯度。
function [f,g] = lr_loss(x,mu,A,b)
[m,n] = size(A);
Ax = A*x;
Atran = A';
expba = exp(- b.*Ax);
f = sum(log(1 + expba))/m + mu*norm(x,2)^2;

if nargout > 1
    g = Atran*(b./(1+expba) - b)/m + 2*mu*x;
end
end
%%%
% 目标函数的海瑟矩阵。要求提供当前优化变量 $x$ 和方向 $u$，返回海瑟矩阵在点 $x$处作用在方向 $u$上的值，即 $\nabla^2 f(x)[u]$。
function H = lr_hess(x,u,A,b,mu)
[m,n] = size(A);
Ax = A*x;
Atran = A';
expba = exp(- b.*Ax);
p = 1./(1 + expba);
w = p.*(1-p);
H = Atran*(w.*(A*u))/m + 2*mu*u;
end
%% 结果分析
% 上图展示了在 LIBSVM 网站中的三个不同的数据集上牛顿法求解过程中梯度范数随迭代步的变化信息。
% 图中的曲线随着迭代步增大斜率的绝对值逐渐增大，这与牛顿法在精确解附近的超线性（二次）收敛性相吻合。
%% 参考页面
% 对于建立的逻辑回归模型，我们调用了牛顿-共轭梯度法对模型进行求解，算法详见
% <fminNewton.html 牛顿-共轭梯度法>。
%
% 在此页面中，我们使用了 LIBSVM 数据集，关于数据集，请参考
% <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ LIBSVM 数据集>。
%
% 此页面的源代码请见： <../download_code/newton/demo_lr_Newton.m demo_lr_Newton.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将