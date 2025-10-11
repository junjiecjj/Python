%% 本程序主要研究多目标情况下不同CFAR算法关于信噪比的检测性能
clc;clear;close all;

%% 参数初始化
N = 64; %滑动窗长度
n = N / 2; %前后沿参考单元长度
Pfa = 1e-6; %虚警概率
SNR_dB = 1:1:35; %信噪比
SNR = 10 .^ (SNR_dB / 10);
Lth = length(SNR); %仿真信噪比区间长度
monte_num = 1e4; % 蒙特卡洛仿真次数，值越大越平滑
rate = 3 / 4;
k = 3 * N / 4; % 有序选择序号
tm_rate1 = 1 / 32;
tm_rate2 = 1 / 16;
cmld_rate = 1 / 4;

%% 门限系数估计
T_CA = ca_threhold(Pfa,N); %CA-CFAR门限因子
T_GO = go_threhold(Pfa,N); %GO-CFAR门限因子
T_SO = so_threhold(Pfa,N); %SO-CFAR门限因子
T_OS = os_threhold(Pfa,N,rate); % OS-CFAR 的门限因子
T_TM = tm_threhold(Pfa,N,tm_rate1,tm_rate2); % TM-CFAR的门限因子
T_CMLD = cmld_threhold(Pfa,N,cmld_rate); % CMLD-CFAR的门限因子

% 检测概率矩阵
Pd_CA = zeros(1, Lth);
Pd_SO = zeros(1, Lth);
Pd_GO = zeros(1, Lth);
Pd_OS = zeros(1, Lth);
Pd_TM = zeros(1, Lth);
Pd_CMLD = zeros(1, Lth);

%% 多目标程序（此处设置了3个目标，可根据自己的需求改）
for i = 1:Lth
    detect_ca = 0; %设置CA检测信号数目变量
    detect_so = 0; %设置SO检测信号数目变量
    detect_go = 0; %设置GO检测信号数目变量
    detect_os = 0; %设置OS检测信号数目变量
    detect_tm = 0; %设置TM检测信号数目变量
    detect_cmld = 0; %设置CMLD检测信号数目变量
    for j = 1:monte_num
        lambda = 1;
        u = rand(1,N);
        exp_noise = log(u) * (-lambda);
        lambda = SNR(i) + 1;
        u = rand(1,3);  % 产生目标
        exp_target = log(u(1)) * (-lambda);  % 目标
        
% 注意 此处可以根据需求选择干扰的位置放在哪里 和 控制干扰的数目 使用时注释掉其他部分
%=================================================================================
        InterPos = randi([1+n,N],1,2); %后沿参考窗产生两个干扰
        exp_noise(InterPos(1)) =  log(u(2)) * (-lambda); 
        exp_noise(InterPos(2)) =  log(u(3)) * (-lambda); 
%=================================================================================
%         InterPos = randi([1,n],1,2); %前沿参考窗产生两个干扰
%         exp_noise(InterPos(1)) =  log(u(2)) * (-lambda); 
%         exp_noise(InterPos(2)) =  log(u(3)) * (-lambda); 
%=================================================================================
%         InterPos1 = randi([1,n],1,1); %前沿参考窗产生1个干扰
%         InterPos2 = randi([1+n,N],1,1); %后沿参考窗产生1个干扰
%         exp_noise(InterPos(1)) =  log(u(2)) * (-lambda); 
%         exp_noise(InterPos(2)) =  log(u(3)) * (-lambda); 
%=================================================================================
        
        % 不同检测器的估计
        % CA-CFAR
        cfar_ca = exp_target / mean(exp_noise);
        if cfar_ca > T_CA
            detect_ca = detect_ca + 1;
        end
        
        % GO-CFAR
        cfar_go = exp_target / max(mean(exp_noise(1:N/2)),mean(exp_noise(N/2+1:N)));
        if cfar_go > T_GO
            detect_go = detect_go + 1;
        end
        
        % SO-CFAR
        cfar_so = exp_target / min(mean(exp_noise(1:N/2)),mean(exp_noise(N/2+1:N)));
        if cfar_so > T_SO
            detect_so = detect_so + 1;
        end
        
        % OS-CFAR
        os_noise = sort(exp_noise);
        cfar_os = exp_target / os_noise(k);
        if cfar_os > T_OS
            detect_os = detect_os + 1;
        end  
        
        % TM-CFAR
        tm_noise = sort(exp_noise);
        tm_noise(1:N*tm_rate1) = [];
        tm_noise(end+1-N*tm_rate2:end) = [];
        cfar_tm = exp_target / mean(tm_noise);
        if cfar_tm > T_TM
            detect_tm = detect_tm + 1;
        end
        
        % CMLD-CFAR
         cmld_noise = sort(exp_noise);
         cmld_noise(end-cmld_rate*N+1:end) = [];
         cfar_cmld = exp_target / mean(cmld_noise);
         if cfar_cmld > T_CMLD
            detect_cmld = detect_cmld + 1;
        end
        
        
    end
    Pd_CA(1,i) = detect_ca / monte_num;
    Pd_SO(1,i) = detect_so / monte_num;
    Pd_GO(1,i) = detect_go / monte_num;
    Pd_OS(1,i) = detect_os / monte_num;
    Pd_TM(1,i) = detect_tm / monte_num;
    Pd_CMLD(1,i) = detect_cmld / monte_num;
end

plot(SNR_dB,Pd_CA,'Color','#FFD700','LineWidth',2,'Marker','<');hold on;   
plot(SNR_dB,Pd_GO,'Color','#DC143C','LineWidth',2,'Marker','+');hold on;   
plot(SNR_dB,Pd_SO,'Color','#FF1493','LineWidth',2,'Marker','*');hold on;   
plot(SNR_dB,Pd_OS,'Color','#8B008B','LineWidth',2,'Marker','v');hold on;   
plot(SNR_dB,Pd_TM,'Color','#483D8B','LineWidth',2,'Marker','^');hold on;   
plot(SNR_dB,Pd_CMLD,'Color','#0000FF','LineWidth',2,'Marker','o');hold on;   

grid minor;
xlabel('\fontname{Times New Roman}SNR/dB');
ylabel('\fontname{宋体}检测概率\fontname{Times New Roman}Pd');
h = legend('CA-CFAR','GO-CFAR','SO-CFAR','OS-CFAR','TM-CFAR','CMLD-CFAR','Location','SouthEast','NumColumns',1);
set(h,'edgecolor','none');




