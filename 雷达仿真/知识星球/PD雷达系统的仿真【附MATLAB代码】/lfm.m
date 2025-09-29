function s=lfm(Pt,Tp,Fr,B,Fs,G,Num_Tr_CPI)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     数据流仿真方式     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s=cell(1,length(Fr));    %发射信号
K=B/Tp;                          % 调频率【Hz】
Tr=1./Fr;                         % 脉冲重复周期【秒】
Delta_t=1/Fs;                    % 时域采样点时间间隔【秒】

Delta_t=1/Fs;                    % 时域采样点时间间隔【秒】
t_set_Tp=0:Delta_t:Tp;                      % 一个脉冲内的时间采样点数组
s_lfm_Tp=sqrt(Pt)*exp(j*pi*K*(t_set_Tp-Tp/2).^2);    % 脉冲信号

N=length(s_lfm_Tp);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for m=1:length(Fr)
s_lfm=zeros(round(Tr(m)*Fs),Num_Tr_CPI); %不同脉冲重复频频率点数向量
for n=1:Num_Tr_CPI
    
     s_lfm(1:N,n)=s_lfm_Tp;
    
end

s{1,m}=s_lfm;
end;
