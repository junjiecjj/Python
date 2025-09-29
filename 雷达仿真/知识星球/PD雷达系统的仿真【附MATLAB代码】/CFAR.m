function [ S_out] = CFAR(S_Sigma_a,Num_Tr_CPI )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%   CFAR�����龯��⣩  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S_out=cell(1,3);
for n_prf=1:length(S_Sigma_a)
 

s_Sigma_a=S_Sigma_a{1,n_prf};
S_output=zeros(size(s_Sigma_a));
Num_sample=size(s_Sigma_a,1);%�洢ά����
Pfa=1e-6;
% �龯����
No_dopper_channel_set=1:Num_Tr_CPI;
% ����CFAR���Ķ�����ͨ�����
Num_ReservedCell=3;
% ���ⵥԪ�����ı�����Ԫ���ȣ�ǰ���
Num_TestWin=50;
% ���ⵥԪ������ͳ���Ӳ��������ʵĴ��ڳ��ȣ�ǰ���
N1=Num_TestWin+Num_ReservedCell;
m=1;
for No_dopper_channel=No_dopper_channel_set
    n=1;
    for No_tr=1+N1:Num_sample-N1%�ӱ�����Ԫ��ĵ�Ԫ��ʼ��⣬���53����ԪҲ�����м�⡣
        Tranning_set=[(No_tr-N1:No_tr-Num_ReservedCell-1),...
            (No_tr+N1:No_tr+Num_ReservedCell+1)];
        % ѵ���������
        Power_noise_clutter=mean(abs(s_Sigma_a(Tranning_set,No_dopper_channel)).^2);
        % ͳ���Ӳ���������
        Threshold=sqrt(2*Power_noise_clutter*log(1/Pfa));
        % �������
        if abs(s_Sigma_a(No_tr,No_dopper_channel))>=Threshold
            % �����о�
            S_output(n,m)=abs(s_Sigma_a(n,m));
        else
            S_output(n,m)=0;
        end
        n=n+1;
    end
    m=m+1;
end
S_out{1,n_prf}=S_output;%S_output��S_out{1,n_prf}ά����ͬ����Num_sample*64
end

end

