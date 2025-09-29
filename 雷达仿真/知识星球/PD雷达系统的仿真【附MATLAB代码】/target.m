function [s_A s_B] = target(G,Fc,Fs,Fr,Num_Tr_CPI,Theta0,Wa,BeamWidth,s,R_set,V_set,RCS,Theta_target_set)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      Ŀ��������     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tr=1./Fr;
c=3e8;  % ����

s_A=cell(1,length(Fr));
s_B=cell(1,length(Fr));


Wavelength=c/Fc;

RCS_Ground_0=10^(-30/10);         % ���浥λ�������ɢ��������m^2��
   
for n_prf=1:length(Fr)
 s_n=s{1,n_prf};   
 s_all_A=zeros(size(s{1,n_prf}));%����Ŀ��ز�
 s_all_B=zeros(size(s{1,n_prf}));%����Ŀ��ز�


s_singal_A=zeros(size(s{1,n_prf}));%����Ŀ��ز�
s_singal_B=zeros(size(s{1,n_prf}));%����Ŀ��ز�

for No_PRI=1:Num_Tr_CPI
    
    Theta_bp=Theta0+Wa*No_PRI*Tr(n_prf);             % ��������ָ�򡾶ȡ�
    
        
        
        delay_target=2*(R_set-V_set*No_PRI*Tr(n_prf))/c;
        delay_ambiguity=mod(delay_target,Tr(n_prf));
        % Ŀ��ʱ�ӡ�sec��
        RVP=-2*pi*Fc*delay_target;%1*3 double
        % Ŀ��ز���Ƶ�첨ʣ����λ
        Gt_A=G*(sinc((Theta_target_set-(Theta_bp-BeamWidth/2))/BeamWidth)).^2;
        Gr_A=G*(sinc((Theta_target_set-(Theta_bp-BeamWidth/2))/BeamWidth)).^2;
        % ����A��Ŀ�귽���ϵ�����
        Gt_B=G*(sinc((Theta_target_set-(Theta_bp+BeamWidth/2))/BeamWidth)).^2;
        Gr_B=G*(sinc((Theta_target_set-(Theta_bp+BeamWidth/2))/BeamWidth)).^2;
        % ����B��Ŀ�귽���ϵ�����
        Lp=1./((4*pi)^2*R_set.^4);
        % Ŀ�괦�ĵ�Ų��������
        Magnitude_echo_A=sqrt(RCS*(Gt_A+Gt_B)*Gr_A*Wavelength^2/(4*pi).*Lp);%1*3 double
        % ����A��Ŀ��ز�����
        Magnitude_echo_B=sqrt(RCS*(Gt_A+Gt_B)*Gr_B*Wavelength^2/(4*pi).*Lp);%1*3 double
        % ����B��Ŀ��ز�����
       
       Echo_delay=round(delay_ambiguity*Fs);  %�ӳٵ��� 

       s_singal_A(:,No_PRI)=circshift(s_n(:,No_PRI),[Echo_delay 0])...%ѭ����λ
           *Magnitude_echo_A(n_prf)*exp(j*RVP(n_prf));%�ӻز����Ⱥ� �Ⲩ��λ��
       %��s_n�ĸ���������λEcho_delay�����з��򲻱䡣
       s_singal_B(:,No_PRI)=circshift(s_n(:,No_PRI),[Echo_delay 0])...
           *Magnitude_echo_B(n_prf)*exp(j*RVP(n_prf));%�ӻز����Ⱥ� �Ⲩ��λ��
        % ����n��Ŀ��ĸ��ز�����ӵ������ź������еĶ�Ӧλ�ã�
end
s_all_A=s_all_A+s_singal_A;%���Ŀ�����
s_all_B=s_all_B+s_singal_B;%���Ŀ�����



s_A{1,n_prf}=s_all_A;
s_B{1,n_prf}=s_all_B;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

