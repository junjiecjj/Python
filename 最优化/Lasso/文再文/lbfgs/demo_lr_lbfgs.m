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


%% 实例：利用 L-BFGS 方法求解逻辑回归问题
% 考虑逻辑回归问题
%
% $$ \displaystyle\min_x \frac{1}{m}\sum_{i=1}^m \log(1+ \exp(-b_ia_i^Tx)) 
% + \mu\|x\|_2^2,$$
% 
% 其中 $(a_i,b_i)_{i=1}^m$ 为已知的待分类的数据集。这里利用 L-BFGS 方法对其进行求解。
% 
%% 逻辑回归问题
%%%
% 设定随机种子。
clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%%
% 在不同的数据集上进行实验。导入 LIBSVM 数据集 a9a 进行实验， |libsvmread| 
% 为另外运行的读入程序。
dataset = 'a9a.test';
[b,A] = libsvmread(dataset);
[m,n] = size(A);
mu = 1e-2/m;
fun = @(x) lr_loss(A,b,m,x,mu);
%%%
% 参数值的设定，|opts.m| 为 L-BFGS 算法的记忆对存储数目。
opts = struct();
opts.xtol = 1e-6;
opts.gtol = 1e-6;
opts.ftol = 1e-16;
opts.maxit = 2000;
opts.record  = 0;
opts.m = 5;
%%%
% 以 $x^0=\mathbf{0}$ 为迭代初始点，调用 L-BFGS 算法求解。
x0 = zeros(n,1);
[x1,~,~,out1] = fminLBFGS_Loop(x0,fun,opts);

%%%
% 在 CINA 上的实验。
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
dataset = 'CINA.test';
[b,A] = libsvmread(dataset);
Atran = A';
[m,n] = size(A);
fun = @(x) lr_loss(x,mu);
x0 = zeros(n,1);
fun = @(x) lr_loss(A,b,m,x,mu);
[x2,~,~,out2] = fminLBFGS_Loop(x0,fun,opts);

%%%
% 在 ijcnn1 上的实验。
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
dataset = 'ijcnn1.test';
[b,A] = libsvmread(dataset);
Atran = A';
[m,n] = size(A);
mu = 1e-2/m;
fun = @(x) lr_loss(A,b,m,x,mu);
x0 = zeros(n,1);
[x3,~,~,out3] = fminLBFGS_Loop(x0,fun,opts);

%% 结果可视化
% 对于不同数据集，展示目标函数的梯度范数随着迭代步的变化。
fig = figure;
k1 = 1:10:out1.iter;
semilogy(k1-1, out1.nrmG(k1), '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
k2 = 1:10:out2.iter;
semilogy(k2-1, out2.nrmG(k2), '-.', 'Color',[0.99 0.1 0.2], 'LineWidth',1.8);
hold on
k3 = 1:10:out3.iter;
semilogy(k3-1, out3.nrmG(k3), '--', 'Color',[0.99 0.1 0.99], 'LineWidth',1.5);
legend('a9a','CINA','ijcnn1');
ylabel('$\|\nabla \ell (x^k)\|_2$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','lr_lbfgs.eps');
%% 辅助函数
% 逻辑回归的目标函数 
% $\frac{1}{m}\sum_{i=1}^m \log(1+ \exp(-b_ia_i^Tx)) + \mu\|x\|_2^2$。
function [f,g] = lr_loss(A,b,m,x,mu)
Ax = A*x;
Atran = A';
expba = exp(- b.*Ax);
f = sum(log(1 + expba))/m + mu*norm(x,2)^2;
%%%
% 当前点处的梯度 $\frac{1}{m}\sum_{i=1}^m -b_ia_i/(1+\exp(b_ia_i^Tx))+2\mu x$。 
% （ |nargout| 表示当前函数在被调用时，需要的输出的个数。当输出个数大于1时，计算目标函数的梯度。）
if nargout > 1
   g = Atran*(b./(1+expba) - b)/m + 2*mu*x;
end
end
%% 结果分析
% 上图展示了在 LIBSVM 的三个数据集上的结果。L-BFGS
% 算法相较牛顿法需要的迭代步数更多，但是不需要在每一步计算海瑟矩阵。
% 另外，在不同的数据集上该算法的表现呈现出较大差异，这与数据集本身的特点有关。
%% 参考页面
% L-BFGS 算法，参见 <fminLBFGS_Loop.html L-BFGS 求解优化问题>。
% 该算法的另一个应用参考页面 <demo_bp_lbfgs.html 实例：L-BFGS 求解基追踪问题>。
% 
% 此页面使用了 LIBSVM 数据集，关于数据集，请参考
% <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ LIBSVM 数据集>。
%
% 此页面的源代码请见：
% <../download_code/lbfgs/demo_lr_lbfgs.m demo_lr_lbfgs.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将
