function s=lfm(Pt,Tp,Fr,B,Fs,G,Num_Tr_CPI)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     ���������淽ʽ     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s=cell(1,length(Fr));    %�����ź�
K=B/Tp;                          % ��Ƶ�ʡ�Hz��
Tr=1./Fr;                         % �����ظ����ڡ��롿
Delta_t=1/Fs;                    % ʱ�������ʱ�������롿

Delta_t=1/Fs;                    % ʱ�������ʱ�������롿
t_set_Tp=0:Delta_t:Tp;                      % һ�������ڵ�ʱ�����������
s_lfm_Tp=sqrt(Pt)*exp(j*pi*K*(t_set_Tp-Tp/2).^2);    % �����ź�

N=length(s_lfm_Tp);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for m=1:length(Fr)
s_lfm=zeros(round(Tr(m)*Fs),Num_Tr_CPI); %��ͬ�����ظ�ƵƵ�ʵ�������
for n=1:Num_Tr_CPI
    
     s_lfm(1:N,n)=s_lfm_Tp;
    
end

s{1,m}=s_lfm;
end;
