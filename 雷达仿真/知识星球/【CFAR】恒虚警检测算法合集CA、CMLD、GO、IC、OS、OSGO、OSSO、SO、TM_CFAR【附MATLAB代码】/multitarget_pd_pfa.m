clc;clear;close all;
% 本程序主要研究多目标情况下不同CFAR算法关于信噪比的检测性能

%% 参数初始化
N = 64; %滑动窗长度
n = N / 2; %前后沿参考单元长度
Pfa = [1e-6:2e-6:9e-6, 1e-5:2e-5:9e-5,1e-4:2e-4:9e-4,1e-3:2e-3:9e-3,1e-2:2e-2:9e-2,1e-1:2e-2:9e-2]; %虚警概率
SNR_dB = 15; %信噪比
SNR = 10 .^ (SNR_dB / 10);
Lth = length(Pfa); %仿真信噪比区间长度
monte_num = 1e5;
k = 3 * N / 4; %有序样本选择序号

T_CA = zeros(1, Lth);
T_GO = zeros(1, Lth);
T_SO = zeros(1, Lth);
T_OS = zeros(1, Lth);

for i = 1 : Lth
    T_CA(1,i) = Pfa(i) ^ (-1 / N) - 1; %门限因子
    T_GO(1,i) = Pfa(i) ^ (-1 / N) - 1; %门限因子
    T_SO(1,i) = Pfa(i) ^ (-1 / N) - 1; %门限因子

    syms T_os
    
    g = Pfa(i) - k * nchoosek(N,k) * gamma(k) * gamma(N-k+1+T_os) / gamma(N+T_os +1);
    x = solve(g);
    T_os = double(x);
    T_OS(1,i) = T_os(T_os  == abs(T_os));

end

% 蒙特卡洛仿真
Pd_CA = zeros(1, Lth);
Pd_SO = zeros(1, Lth);
Pd_GO = zeros(1, Lth);
Pd_OS = zeros(1, Lth);

%% 多目标程序
for i = 1:Lth
    detect_ca = 0; % 设置CA检测信号数目变量
    detect_so = 0; % 设置SO检测信号数目变量
    detect_go = 0; % 设置GO检测信号数目变量
    detect_os = 0; % 设置OS检测信号数目变量
    for j = 1:monte_num
        lambda = 1;
        u = rand(1,N-5);
        exp_noise = log(u) * (-lambda);
        lambda = SNR + 1;
        u = rand(1,6);  % 
        exp_target = log(u(1)) * (-lambda);  % 目标
        
        exp_noise(N-4) = log(u(2)) * (-lambda);  % 干扰目标1
        exp_noise(N-3) = log(u(3)) * (-lambda);  % 干扰目标2
        exp_noise(N-2) = log(u(4)) * (-lambda);  % 干扰目标3
        exp_noise(N-1) = log(u(5)) * (-lambda);  % 干扰目标4
        exp_noise(N) = log(u(6)) * (-lambda);  % 干扰目标5
        
        cfar_ca = exp_target / sum(exp_noise);
        if cfar_ca > T_CA(i)
            detect_ca = detect_ca + 1;
        end
        
        cfar_go = exp_target / max(sum(exp_noise(1:N/2)),sum(exp_noise(N/2+1:N)));
        if cfar_go > T_GO(i)
            detect_go = detect_go + 1;
        end
        
        cfar_so = exp_target / min(sum(exp_noise(1:N/2)),sum(exp_noise(N/2+1:N)));
        if cfar_so > T_SO(i)
            detect_so = detect_so + 1;
        end
        
        exp_noise = sort(exp_noise);
        cfar_os = exp_target / exp_noise(k);
        if cfar_os > T_OS(i)
            detect_os = detect_os + 1;
        end  
        
    end
    Pd_CA(1,i) = detect_ca / monte_num;
    Pd_SO(1,i) = detect_so / monte_num;
    Pd_GO(1,i) = detect_go / monte_num;
    Pd_OS(1,i) = detect_os / monte_num;
end

% %% 单目标程序
% for i = 1:Lth
%     detect_ca = 0; %设置CA检测信号数目变量
%     detect_so = 0; %设置SO检测信号数目变量
%     detect_go = 0; %设置GO检测信号数目变量
%     detect_os = 0; %设置OS检测信号数目变量
%     for j = 1:monte_num
%         lambda = 1;
%         u = rand(1,N);
%         exp_noise = log(u) * (-lambda);
%         lambda = SNR + 1;
%         u = rand(1,1);  % 
%         exp_target = log(u(1)) * (-lambda);  % 目标
%         
%         cfar_ca = exp_target / sum(exp_noise);
%         if cfar_ca > T_CA(i)
%             detect_ca = detect_ca + 1;
%         end
%         
%         cfar_go = exp_target / max(sum(exp_noise(1:N/2)),sum(exp_noise(N/2+1:N)));
%         if cfar_go > T_GO(i)
%             detect_go = detect_go + 1;
%         end
%         
%         cfar_so = exp_target / min(sum(exp_noise(1:N/2)),sum(exp_noise(N/2+1:N)));
%         if cfar_so > T_SO(i)
%             detect_so = detect_so + 1;
%         end
%         
%         exp_noise = sort(exp_noise);
%         cfar_os = exp_target / exp_noise(k);
%         if cfar_os > T_OS(i)
%             detect_os = detect_os + 1;
%         end  
%         
%     end
%     Pd_CA(1,i) = detect_ca / monte_num;
%     Pd_SO(1,i) = detect_so / monte_num;
%     Pd_GO(1,i) = detect_go / monte_num;
%     Pd_OS(1,i) = detect_os / monte_num;
% end

semilogx(Pfa,Pd_CA,'rd-','LineWidth',1.5);hold on;   
semilogx(Pfa,Pd_GO,'g-*','LineWidth',1.5);hold on;   
semilogx(Pfa,Pd_SO,'b-.','LineWidth',1.5);hold on;   
semilogx(Pfa,Pd_OS,'k-x','LineWidth',1.5);hold on;   

grid minor;
xlabel('\fontname{宋体}虚警概率');
ylabel('\fontname{宋体}检测概率\fontname{Times New Roman}Pd');
h = legend('CA-CFAR','GO-CFAR','SO-CFAR','OS-CFAR','Location','SouthEast','NumColumns',1);
set(h,'edgecolor','none');




