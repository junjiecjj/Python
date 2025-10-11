clc;clear all;close all;
% 本程序主要研究多目标情况下OS-CFAR关于信噪比的检测性能

%% 参数初始化
N = 36; %滑动窗长度
k = 3 * N / 4; %有序样本选择序号
n = N / 2; %前后沿参考单元长度
Pfa = 1e-6; %虚警概率
SNR_dB = 5:1:35; %信噪比
SNR = 10 .^ (SNR_dB / 10);
Lth = length(SNR); %仿真信噪比区间长度

syms T_os 
g = Pfa - k * nchoosek(N,k) * gamma(k) * gamma(N-k+1+T_os) / gamma(N+T_os+1);
x = solve(g);
T_os = double(x);
T_os = T_os(T_os == abs(T_os));
monte_num = 1e5;

%%蒙特卡洛模拟
Pd_OS1 = 0;
for i = 1:Lth
    detect_num = 0; %设置检测信号数目变量
    for j = 1:monte_num
        lambda = 1;
        u = rand(1,N-2);
        exp_noise = log(u) * (-lambda);
        lambda = SNR(i) + 1;
        u = rand(1,3);
        exp_target = log(u(1)) * (-lambda);
        exp_noise(N-1) = log(u(2)) * (-lambda);
        exp_noise(N) = log(u(3)) * (-lambda);
        exp_noise = sort(exp_noise);
        cfar = exp_target / exp_noise(k);
        if cfar > T_os
            detect_num = detect_num + 1;
        end
    end
    Pd_OS1(i) = detect_num / monte_num;
end
plot(SNR_dB,Pd_OS1,'bo-','LineWidth',1.5);
hold on;

Pd_OS2 = 0;
for i = 1:Lth
    detect_num = 0; %设置检测信号数目变量
    for j = 1:monte_num
        lambda = 1;
        u = rand(1,N-5);
        exp_noise = log(u) * (-lambda);
        lambda = SNR(i) + 1;
        u = rand(1,6);
        exp_target = log(u(6)) * (-lambda);
        exp_noise(N-4) = log(u(5)) * (-lambda);
        exp_noise(N-3) = log(u(4)) * (-lambda);
        exp_noise(N-2) = log(u(3)) * (-lambda);
        exp_noise(N-1) = log(u(2)) * (-lambda);
        exp_noise(N) = log(u(1)) * (-lambda);
        exp_noise = sort(exp_noise);
        cfar = exp_target / exp_noise(k);
        if cfar > T_os
            detect_num = detect_num + 1;
        end
    end
    Pd_OS2(i) = detect_num / monte_num;
end
plot(SNR_dB,Pd_OS2,'rs-','LineWidth',1.5);
hold on;        

Pd_OS3 = 0;
for i = 1:Lth
    detect_num = 0; %设置检测信号数目变量
    for j = 1:monte_num
        lambda = 1;
        u = rand(1,N-10);
        exp_noise = log(u) * (-lambda);
        lambda = SNR(i) + 1;
        u = rand(1,11);
        exp_target = log(u(11)) * (-lambda);
        exp_noise(N-9) = log(u(10)) * (-lambda);
        exp_noise(N-8) = log(u(9)) * (-lambda);
        exp_noise(N-7) = log(u(8)) * (-lambda);
        exp_noise(N-6) = log(u(7)) * (-lambda);
        exp_noise(N-5) = log(u(6)) * (-lambda);
        exp_noise(N-4) = log(u(5)) * (-lambda);
        exp_noise(N-3) = log(u(4)) * (-lambda);
        exp_noise(N-2) = log(u(3)) * (-lambda);
        exp_noise(N-1) = log(u(2)) * (-lambda);
        exp_noise(N) = log(u(1)) * (-lambda);
        exp_noise = sort(exp_noise);
        cfar = exp_target / exp_noise(k);
        if cfar > T_os
            detect_num = detect_num + 1;
        end
    end
    Pd_OS3(i) = detect_num / monte_num;
end
plot(SNR_dB,Pd_OS3,'kd-','LineWidth',1.5);
hold on;     

Pd_OS3 = 0;
for i = 1:Lth
    detect_num = 0; %设置检测信号数目变量
    for j = 1:monte_num
        lambda = 1;
        u = rand(1,N-10);
        exp_noise = log(u) * (-lambda);
        lambda = SNR(i) + 1;
        u = rand(1,11);
        exp_target = log(u(11)) * (-lambda);
        exp_noise(N-9) = log(u(10)) * (-lambda);
        exp_noise(N-8) = log(u(9)) * (-lambda);
        exp_noise(N-7) = log(u(8)) * (-lambda);
        exp_noise(N-6) = log(u(7)) * (-lambda);
        exp_noise(N-5) = log(u(6)) * (-lambda);
        exp_noise(N-4) = log(u(5)) * (-lambda);
        exp_noise(N-3) = log(u(4)) * (-lambda);
        exp_noise(N-2) = log(u(3)) * (-lambda);
        exp_noise(N-1) = log(u(2)) * (-lambda);
        exp_noise(N) = log(u(1)) * (-lambda);
        exp_noise = sort(exp_noise);
        cfar = exp_target / exp_noise(k);
        if cfar > T_os
            detect_num = detect_num + 1;
        end
    end
    Pd_OS3(i) = detect_num / monte_num;
end
plot(SNR_dB,Pd_OS3,'kd-','LineWidth',1.5);
hold on;   
grid minor;
xlabel('\fontname{Times New Roman}SNR/dB');
ylabel('\fontname{宋体}检测概率\fontname{Times New Roman}Pd');
title('\fontname{Times New Roman}r\fontname{宋体}个干扰目标情况下\fontname{Times New Roman}OS-CAFR\fontname{宋体}检测概率\fontname{Times New Roman}(N=36,Pfa=1e-6)');
h = legend('r=2','r=5','r=10','Location','SouthEast','NumColumns',1);
set(h,'edgecolor','none');