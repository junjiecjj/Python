
% This is the reproduction code for our JSTSP paper.
% "Simultaneous Sensing Data Acquisition and Sharing in Low-Altitude Wireless Networks: Fundamental Limits and Signaling Design"
% Specifically, this code shows the process of the optimal channel input distribution design, corresponding Section IV.
% 计算分为两个部分：第一部分先计算给定感知失真Ds时，求得带有失真和功率约束的通信信道容量C
% 第二部分以信道容量C根据率失真函数求得通信传输过程中的失真Dc，最终失真为Ds+Dc
%% By Fuwang Dong 2024/05/05
clear;
close all;
clc;
%%  System Setting
N_symbol            =  251;                                       % 总发射符号
Power_B             =  5;                                         % 输出信号功率
%  输入输出码本和符号集合
X_input             =  linspace(-10,10,N_symbol).';               % 感知和通信信道输入码本
Y_output            =  linspace(-10,10,N_symbol).';               % 通信侧接收到的信号码本
S                   =  linspace(-3,3,N_symbol).';                 % 目标状态参数
hatS_BS             =  linspace(-3,3,N_symbol).';                 % 基站对目标参数估计（感知）
hatS_UE             =  linspace(-3,3,N_symbol).';                 % 用户对目标参数估计（通信）

%%  符号的先验PDF，这里假设通感信道均为N(0,1)的高斯信道 
%   假设状态参数服从N(0,1)高斯分布
sigma_s             =  1;
P_S                 =  1./sqrt(2*pi*sigma_s)*exp(-S.^2/(2*sigma_s));              % 参数S服从Gaussian分布N(0,1)
P_S                 =  P_S ./ sum(P_S);                            % 归一化使其成为PDF

%% 定义初始对偶变量参数和cost与失真函数
b                   =  @(x) x.^2;                                  % 功率约束cost
% 估计的MMSE，这篇论文的估计实际上就是MMSE估计器，因此失真函数就是给定x的情况下，z对参数s的MMSE估计对所有的s和z积分，因此失真是x的函数。
% 这里考虑线型模型Z=XS+N, 在给定X的情况下，对hasS_BS的估计
% MMSE完整公式为 sigma_n2*sigma_s2/(snr*sigma_s2+sigma_n2)，这里sigma_s=1,snr=x^2
c                   =  @(x) 1./(1+x.^2);   


%% 第一部分，计算感知失真Ds与通信信道容量C

Iteration_BA        =  50;                                          % BA算法外部迭代次数 
Mu_set              =  [0:0.1:1,1:0.5:5,5:5:30];                    % 不同参数调整D_s和MI之间的关系
N_DsCap             =  length(Mu_set);
Capacity            =  zeros(1,N_DsCap);
D_s                 =  zeros(1,N_DsCap);
D_c                 =  zeros(1,N_DsCap);
D                   =  zeros(1,N_DsCap);
PX_set              =  zeros(N_symbol,N_DsCap);

for  iter_mu   =  1:N_DsCap

    mu          =  Mu_set(iter_mu);

    % 计算信道输入的主程序

    P_X         =  1/N_symbol*ones(N_symbol,1);               % 初始信道输入为均匀分布    
    for iter = 1:Iteration_BA
        
        Q_Y_given_X              =  zeros(N_symbol,N_symbol);   % 初始化Y|X的条件概率密度矩阵
    
        % 这里考虑通信的信道模型为Y=X+N，给定X时，Y为服从N(X,1)的Gaussian分布
        for  ix  =  1:N_symbol
             Q_Y_given_X(:,ix)   =  1./sqrt(2*pi)*exp(-(Y_output-X_input(ix)).^2/2);                       
        end
        % 归一化概率，使其成为PDF 
        for  ix  =  1:N_symbol
             temp                =  sum(Q_Y_given_X(:,ix));
             Q_Y_given_X(:,ix)   =  Q_Y_given_X(:,ix)/temp;                       
        end

        
        P_X_matrix               =  repmat(P_X.',N_symbol,1);
        P_YX                     =  Q_Y_given_X.*P_X_matrix;       % 计算P(X,Y)=Q(Y|X)*P(X) 
        P_Y                      =  sum(P_YX,2);                   % 计算P(Y)=sum_X Q(Y|X)*P(X)   
        Q_X_given_Y              =  P_YX./P_Y;                     % 计算Q(X|Y)
    
        %  计算Lagrange函数g(x)
        gx_temp                  =  Q_Y_given_X.*log(Q_X_given_Y);
        gx                       =  sum(gx_temp).';
        [P_X,lambda,cost]        =  SF_Update_PX_lambda(mu,X_input,gx,b,c,Power_B);    %内部循环搜索满足条件的lambda，并且更新PX 

    end

    I_XY                =  P_YX.*log2(Q_Y_given_X./P_Y);
    I_XY(isnan(I_XY))   =  0;
    Cap_u               =  sum(sum(I_XY));
    MMSE                =  P_X'*c(X_input); 
    Capacity(iter_mu)   =  Cap_u;
    D_s(iter_mu)        =  MMSE;
    PX_set(:,iter_mu)   =  P_X;
    % 输出
    disp(['Capacity: ' num2str(Cap_u)]);
    disp(['MMSE: ' num2str(MMSE)]);

%%  第二部分，计算给定C时的率失真函数RD，得到通信失真Dc
    P_hatS_BS                =  SF_MMSE_Distribution(P_X,X_input,hatS_BS,sigma_s);                  %求得基站估计hatS的分布
%     plot(hatS_BS, P_hatS_BS,'b')
    Slope                    =  -10:0.5:-2;                                                 %率失真函数的斜率
    [Rate_S, Distortion_S]   =  SF_BA_Rate_Distortion(hatS_BS, hatS_UE, Slope, P_hatS_BS);  %计算hatS_BS和hatS_UE之间的率失真函数
%     plot(Distortion_S, Rate_S,'r')
    RD                       =  polyfit(Rate_S,Distortion_S,3);                             % 曲线拟合画出RD函数
    Dc_u                     =  polyval(RD,Cap_u);                                          % 计算Dc
    D_c(iter_mu)             =  Dc_u;
    D(iter_mu)               =  MMSE+Dc_u;                 
end

%%%%%%%%%%%%%%%%     存储需要的数据  %%%%%%%%%%%%%%%%%

% filename = strcat('Optimal_Channel_Input','.mat');
% save(filename,'D_c','D_s', 'D', 'Capacity', 'PX_set','X_input');

%% Figure
%  颜色与线型选择

colorset_str      =  {'#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295'};    % 线条颜色集合
colorset_rgb      =  zeros(length(colorset_str),3);       

%  将颜色转化为Matlab认识的rgb值
clist    = '0123456789ABCDEF';

for nn   =  1:length(colorset_str)

    str      =  colorset_str{nn};
    nums     =  zeros(1, 6);
    for i = 1:6
        nums(i) = find(str(i + 1) == clist);
    end
    rgb = zeros(1, 3);
    for i = 1:3
        rgb(i) = 16 * (nums(2*i-1) - 1) + (nums(2*i) - 1);
    end

    colorset_rgb(nn,:)  =  round(100 * rgb/255) / 100; 
end

styleset    =  ["-";"--";":";"-."];
markerset   =  ["none";"o";"+";"*";"x";"s";"p";"d";"^";"v";"<";">";"h"];


figure (1)
colororder([colorset_rgb(1,:); colorset_rgb(6,:)])

yyaxis left
plot(Capacity, D_s,'LineWidth',1.5,'LineStyle',styleset(1),'Marker',markerset(2),'MarkerSize',5,'Color', colorset_rgb(1,:))
hold on 
grid on
xlabel('Communication Channel Capacity','interpreter','latex','FontSize',14,'fontname','Times New Roman')
ylabel('Sensing Disotortion $D_s$','interpreter','latex','FontSize',14) 

yyaxis right
plot(Capacity, D_c,'LineWidth',1.5,'LineStyle',styleset(1),'Marker',markerset(4),'MarkerSize',5,'Color', colorset_rgb(6,:))
hold on 
grid on
xlabel('Communication Channel Capacity','interpreter','latex','FontSize',14,'fontname','Times New Roman')
ylabel('Communication Disotortion $D_c$','interpreter','latex','FontSize',14)

figure (2)
plot(Capacity, D,'LineWidth',1.5,'LineStyle',styleset(2),'Marker',markerset(3),'MarkerSize',5,'Color', colorset_rgb(1,:))
hold on 
grid on
xlabel('Communication Channel Capacity','interpreter','latex','FontSize',14,'fontname','Times New Roman')
ylabel('SDAS Disotortion','interpreter','latex','FontSize',14) 


                

    



























