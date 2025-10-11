clear; clc; 
rng(966)
% N = [10; 100; 1000];% 待优化变量维度
N = 2000;
Time = zeros(length(N), 8);% 计算8条曲线的用时

Nit = 1e3;
%%
for i_N = 1:length(N)
    %% 生成数据A和x
    A = randn(N(i_N));
    A = A'*A;% 半正定
    x_ini = randn(N(i_N), 1);% 初始随机变量
    %归一化
    x_ini = x_ini/norm(x_ini);
    A = A/norm(A, 'fro');
    %% 固定步长(3种不同的步长)
    step = [2; 1; 1e-1; 1e-2];
    f_Fixed = zeros(Nit, length(step));
    for i = 1:length(step)
        tic
        f_Fixed(:, i) = GD_FixedStep(Nit, A, x_ini, step(i));
        Time(i_N, i) = toc;        
    end
    %% Armijo-Goldstein准则
    rho = 5e-1;% 将某点的一阶泰勒展开对应的直线拉到偏水平一些
    alpha_ini = [2, 5, 5];
    
    tic
    f_AG = GD_AG(Nit, rho, alpha_ini(i_N), A, x_ini);
    Time(i_N, 5) = toc; 
    %% Wolfe-Powell准则
    rho = 5e-1;% 将某点的一阶泰勒展开对应的直线拉到偏水平一些
    alpha_ini = [2, 5, 5];
    sigma = 0.5;
    x = x_ini;
    count = 1;
    count_max = 20;
    const = 1.2;
    
    tic
    f_WP = GD_WP(Nit, rho, alpha_ini(i_N), A, x, sigma, count, count_max, const);
    Time(i_N, 6) = toc;
    %% 最优步长(对步长求导=0)
    tic
    f_opt = GD_Opt(Nit, A, x_ini);
    Time(i_N, 7) = toc;
    %% Zhuoran准则(自编)
    rho = 5e-1;% 将某点的一阶泰勒展开对应的直线拉到偏水平一些
    sigma = [0.01 0.5 0.5];
    count = 1;
    count_max = 20;
    alpha_ini = [1 1 2];
    
    tic
    f_ZR = GD_ZR(Nit, rho, alpha_ini(i_N), sigma(i_N), count, count_max, A, x_ini);
    Time(i_N, 8) = toc;
    %% 绘图
    set(0,'defaultfigurecolor','w') 
    figure; hold on; box on; grid on;
    set(gca,'FontSize',10);
    xlabel('迭代次数');
    ylabel('log_{10}(目标函数)');
    text = ['优化变量维度：N = ' num2str(N(i_N))];
    title(text)
    plot(log10(f_Fixed(:, 1)),'r-', 'LineWidth', 1.5,'MarkerSize',10);
    plot(log10(f_Fixed(:, 2)),'r--', 'LineWidth', 1.5,'MarkerSize',10);
    plot(log10(f_Fixed(:, 3)),'r-.', 'LineWidth', 1.5,'MarkerSize',10);
    plot(log10(f_Fixed(:, 4)),'k:', 'LineWidth', 1.5,'MarkerSize',10);
    plot(log10(f_AG),'g:', 'LineWidth', 1.5,'MarkerSize',10);
    plot(log10(f_opt),'m:', 'LineWidth', 1.5,'MarkerSize',10);
    plot(log10(f_WP),'b-', 'LineWidth', 1.5,'MarkerSize',10);
    plot(log10(f_ZR),'m-', 'LineWidth', 1.5,'MarkerSize',10);
    legend({  
        ['固定步长 step=' num2str(step(1)) ' | 用时' num2str(Time(i_N, 1)) 's'],...
        ['固定步长 step=' num2str(step(2)) ' | 用时' num2str(Time(i_N, 2)) 's'],...
        ['固定步长 step=' num2str(step(3)) ' | 用时' num2str(Time(i_N, 3)) 's'],...
        ['固定步长 step=' num2str(step(4)) ' | 用时' num2str(Time(i_N, 4)) 's'],...
        ['Armijo-Goldstein' ' | 用时' num2str(Time(i_N, 5)) 's'],...
        ['初始梯度下的最优步长'  ' | 用时' num2str(Time(i_N, 6)) 's'],...
        ['Wolfe-Powell'  ' | 用时' num2str(Time(i_N, 7)) 's'],...
        ['Zhuoran'  ' | 用时' num2str(Time(i_N, 8)) 's']})
end