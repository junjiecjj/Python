clc;clear all;close all;
% 本程序主要研究多目标情况下CA-CFAR关于信噪比的检测性能
%% 参数初始化
N = 36; %滑动窗长度
n = N / 2; %前后沿参考单元长度
Pfa = 1e-6; %虚警概率
SNR_dB = 5:1:35; %信噪比
SNR = 10 .^ (SNR_dB / 10);
Lth = length(SNR); %仿真信噪比区间长度
T = Pfa ^ (-1 / N) - 1; %门限因子
monte_num = 1e5;

%%蒙特卡洛模拟
Pd_CA1 = 0;
for i = 1:Lth
    detect_num = 0; %设置检测信号数目变量
    for j = 1:monte_num
        lambda = 1;
        u = rand(1,N-1);
        exp_noise = log(u) * (-lambda);
        lambda = SNR(i) + 1;
        u = rand(1,2);
        exp_target = log(u(1)) * (-lambda);
        exp_noise(N) = log(u(2)) * (-lambda);
        cfar = exp_target / sum(exp_noise);
        if cfar > T
            detect_num = detect_num + 1;
        end
    end
    Pd_CA1(i) = detect_num / monte_num;
end
plot(SNR_dB,Pd_CA1,'bo-','LineWidth',1.5);
hold on;

Pd_CA2 = 0;
for i = 1:Lth
    detect_num = 0; %设置检测信号数目变量
    for j = 1:monte_num
        lambda = 1;
        u = rand(1,N-2);
        exp_noise = log(u) * (-lambda);
        lambda = SNR(i) + 1;
        u = rand(1,3);
        exp_target = log(u(3)) * (-lambda);
        exp_noise(N-1) = log(u(2)) * (-lambda);
        exp_noise(N) = log(u(1)) * (-lambda);
        cfar = exp_target / sum(exp_noise);
        if cfar > T
            detect_num = detect_num + 1;
        end
    end
    Pd_CA2(i) = detect_num / monte_num;
end
plot(SNR_dB,Pd_CA2,'rs-','LineWidth',1.5);
hold on;        

Pd_CA3 = 0;
for i = 1:Lth
    detect_num = 0; %设置检测信号数目变量
    for j = 1:monte_num
        lambda = 1;
        u = rand(1,N-3);
        exp_noise = log(u) * (-lambda);
        lambda = SNR(i) + 1;
        u = rand(1,4);
        exp_target = log(u(4)) * (-lambda);
        exp_noise(N-2) = log(u(3)) * (-lambda);
        exp_noise(N-1) = log(u(2)) * (-lambda);
        exp_noise(N) = log(u(1)) * (-lambda);
        cfar = exp_target / sum(exp_noise);
        if cfar > T
            detect_num = detect_num + 1;
        end
    end
    Pd_CA3(i) = detect_num / monte_num;
end

plot(SNR_dB,Pd_CA3,'kd-','LineWidth',1.5);
hold on;     
grid minor;
xlabel('\fontname{Times New Roman}SNR/dB');
ylabel('\fontname{宋体}检测概率\fontname{Times New Roman}Pd');
title('\fontname{Times New Roman}r\fontname{宋体}个干扰目标情况下\fontname{Times New Roman}CA-CAFR\fontname{宋体}检测概率\fontname{Times New Roman}(N=36,Pfa=1e-6)');
h = legend('r=1','r=2','r=3','Location','SouthEast','NumColumns',1);
set(h,'edgecolor','none');