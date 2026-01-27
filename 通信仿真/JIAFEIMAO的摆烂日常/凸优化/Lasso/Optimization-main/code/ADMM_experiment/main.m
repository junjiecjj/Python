%% setting
clc; 
rng(996)
set(0,'defaultfigurecolor','w') 
N_all = [10, 100, 1000];
CompressRatio = 0.5;
M_all = N_all*CompressRatio;
rho = 0.1;
lambda = 0.1;
Nit = 1e2;
Nmean = 5e1;
NMSE = zeros(Nit, length(N_all), Nmean);
for i_mean = 1:Nmean
    for i_N = 1:length(N_all)
        N = N_all(i_N);
        M = M_all(i_N);
        A = randn(M, N);
        x = randn(N, 1);
        z = randn(N, 1);
        u = zeros(N, 1);
        x_real = zeros(N, 1);


        % 生成稀疏的x
        SparseRatio = 0.1;
        NumNoZero = SparseRatio * N;
        tmp = randperm(N);
        x_real(tmp(1:NumNoZero)) = randn(NumNoZero, 1);
        b = A*x_real;
        %% ADMM
        for i_it = 1:Nit
            NMSE(i_it, i_N, i_mean) = norm(x_real - x)^2 / norm(x_real)^2;
            x = (A.'*A + rho*eye(size(A, 2)))\...
                     (A.'*b + rho*(z - u));
            z = wthresh(x + u/rho, 's', lambda/rho);
            u = u + (x - z)*rho;    
        end
        %% 绘图信号恢复情况
        if i_mean == 1
            figure; hold on; box on; grid on;
            set(gca,'FontSize',10);
            xlabel('x的index');
            ylabel('x的取值');
            text = ['优化变量维度：N = ' num2str(N) ' | ' '变量稀疏度：' num2str(SparseRatio)];
            title(text)
            stem(x_real,'b--s', 'LineWidth', 1,'MarkerSize',10)
            stem(x,'g:', 'LineWidth', 1,'MarkerSize',10)
            plot(x-x_real,'m', 'LineWidth', 3,'MarkerSize',10)
            ylim([-4, 4])
            legend({  
                ['真实的稀疏向量x_{real}'],...
                ['ADMM估计的稀疏向量x_{est}'],...
                ['x_{real}-x_{est}']})
        end
    end
end
%% Nmean次求均值
NMSEmean = zeros(size(NMSE, 1), size(NMSE, 2));
for i = 1:size(NMSE, 2)
    NMSEmean(:, i) = sum(squeeze(NMSE(:,i,:)), 2)/Nmean;
end
%% 绘图NMSE
figure; hold on; box on; grid on;
set(gca,'FontSize',10);
xlabel('迭代次数');
ylabel('log_{10}(NMSE)');
text = ['变量稀疏度：' num2str(SparseRatio)];
title(text)
plot(log10(NMSEmean(:, 1)), 'r-', 'LineWidth', 1.5,'MarkerSize',10)
plot(log10(NMSEmean(:, 2)), 'm--', 'LineWidth', 1.5,'MarkerSize',10)
plot(log10(NMSEmean(:, 3)), 'g:', 'LineWidth', 1.5,'MarkerSize',10)
    legend({  
        ['优化变量维度：N = ' num2str(N_all(1)) ],...
        ['优化变量维度：N = ' num2str(N_all(2)) ],...
        ['优化变量维度：N = ' num2str(N_all(3)) ]})