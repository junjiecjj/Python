function [ angle_result ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%       ��������       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
angle_result=cell(1,length(S_Sigma_a));
for n_prf=1:length(S_Sigma_a)
S_Sigma=S_Sigma_a{1,n_prf};
S_Delta=S_Delta_a{1,n_prf};
angle_result{1,n_prf}=zeros(size(S_Sigma));
Theta_set=-BeamWidth/2:BeamWidth/100:BeamWidth/2;
% ���Χ��101���㡣
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%���㲻ͬ�ǶȵĺͲ��ֵ %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size(Theta_set)
n=1;
for Theta=Theta_set
    G_A=G*(sinc((Theta+BeamShift)/BeamWidth)).^2;
    G_B=G*(sinc((Theta-BeamShift)/BeamWidth)).^2;
    % ���㲨��A��B�ķ���ͼ
    Sigma_G=G_A+G_B;
    Delta_G=G_A-G_B;
    % ����͡�����ķ���ͼ
    Delta_2_Sigma_ratio(n)=Delta_G/Sigma_G;
    % ����Ͳ��
    n=n+1;
end



Target_Range_No=Target_Range_all{1,n_prf};
Target_Doppler_No=Target_Doppler_all{1,n_prf};

for No_target=1:length(Target_Range_No)
    Delta_target=S_Delta(Target_Range_No(No_target),Target_Doppler_No(No_target));
    Sigma_target=S_Sigma(Target_Range_No(No_target),Target_Doppler_No(No_target));
%     angle_Sigma=angle(Sigma_target);
%     angle_Delta =angle(Delta_target);
%     % ��Ͳ�ͨ�������ǲ�
%     if abs(angle_Sigma-angle_Delta)>pi/2
%         Sign_Delta_Sigma=-1;
%     else
%         Sign_Delta_Sigma=1;
%     end
%     % ȷ��Ŀ��Ƕ�����
    Delta_2_Sigma_ratio_Target(No_target)=Delta_target/Sigma_target;
    % ��Ͳ��
    [min_v,p_theta]=min(abs(Delta_2_Sigma_ratio_Target(No_target)-Delta_2_Sigma_ratio));
    %�ҳ������ֵ�����С�ı��ֵ��
    Theta_targe(No_target)=Theta_set(p_theta)+Theta0;
    angle_result{1,n_prf}(Target_Range_No(No_target),Target_Doppler_No(No_target))=Theta_targe(No_target)*180/pi;
    disp(Theta_targe(No_target)*180/pi)
    % �����Ŀ��Ƕ�
end
  
end

end

