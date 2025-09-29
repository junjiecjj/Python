clear all
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%    �״�ϵͳ�������    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=3e8;                           % ����
k=1.38e-23;                      % ������������

Pt=20e3;                         % ���书�ʡ�W��

Fc=1e9;                          % ����Ƶ�ʡ�Hz��
Wavelength=c/Fc;                 % ����������m��

Tp=8e-6;                        % �����ȡ�΢�롿
Fr=[8e3 11e3 13e3];                         % �����ظ�Ƶ�ʡ�Hz��

B=10e6;                           % ����Hz��
Fs=20e6;                         % �����ʡ�Hz��
F=10^(6.99/10);                     % ����ϵ��
K=B/Tp;                          % ��Ƶ�ʡ�Hz��
Tr=1./Fr;% �����ظ����ڡ��롿
R_T=Tr*c/2;%���ģ������

Delta_t=1/Fs;                    % ʱ�������ʱ�������롿
vv=Fr*Wavelength/2;  %���ģ���ٶ�
D=5;                             % ���߿׾���m��
Ae=1*pi*(D/2)^2;                 % ������Ч�����m^2��
% G=4*pi*Ae/Wavelength^2;          % ��������
G=10^(32/10);
BeamWidth=0.88*Wavelength/D;     % ����3dB������ȡ�deg��
BeamShift=0.8*BeamWidth/2;         % A��B��������������ļнǡ�deg��
Theta0=30*pi/180;                % ���������ʼָ�򡾶ȡ�
Wa=0;2*pi/1;                       % ���߲���ת�١�rad/sec��

Num_Tr_CPI=64;                      % CPI������


R_set=[70e3,7e3,10e3];          % Ŀ����롾m��         
RCS=[1,1,1];                 % Ŀ��ƽ������ɢ��������m^2��   
Theta_target_set=30.1*pi/180; % Ŀ�귽λ�ǡ�deg��
V_set=[2100,1000,900];                % Ŀ���ٶȡ�m/s�� 

for a=1:length(Fr)
    
   R_A(a)=mod(R_set(1),R_T(a));%�ж��Ƿ����ģ��
end
for a=1:length(Fr)
    
   v_A(a)=mod(V_set(1),vv(a));
end    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      ���������ź�     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s=lfm(Pt,Tp,Fr,B,Fs,G,Num_Tr_CPI);

figure
s_plot(s);
title('�״﷢���ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')
print(gcf,'-dbitmap','F:\����ͼƬ\�״﷢���ź�.bmp')   % ����Ϊpng��ʽ��ͼƬ��


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      Ŀ��ز�     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s_A s_B] = target(G,Fc,Fs,Fr,Num_Tr_CPI,Theta0,Wa,BeamWidth,s,R_set,V_set,RCS,Theta_target_set);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      ����������     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s_A s_B] = nose(s_A,s_B,k,B,F);

figure
subplot(2,1,1)
s_plot(s_A);
title('Aͨ���ز��ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')

subplot(2,1,2)
s_plot(s_B);
title('Bͨ���ز��ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')

print(gcf,'-dbitmap','F:\����ͼƬ\�״�ز��ź�.bmp')   % ����Ϊpng��ʽ��ͼƬ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      �Ͳ������    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[s_Sigma s_Delta] =sigma_delta(s_A,s_B);


figure
subplot(2,1,1)
s_plot(s_Sigma);
title('��ͨ���ز��ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')

subplot(2,1,2)
s_plot(s_Delta);
title('��ͨ���ز��ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')
print(gcf,'-dbitmap','F:\����ͼƬ\�Ͳ���ƻز��ź�.bmp')   % ����Ϊpng��ʽ��ͼƬ��



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  ƥ���˲�������ѹ��)  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%]
[s_Sigma_rc s_Delta_rc] = match(s_Sigma,s_Delta,Tr,Fs,K,Num_Tr_CPI);

figure
s_plot(s_Sigma_rc);
title('��ͨ��ƥ���˲����')
xlabel('time [sec]')
ylabel('magnitude [v]')


print(gcf,'-dbitmap','F:\����ͼƬ\ƥ���˲����.bmp')   % ����Ϊpng��ʽ��ͼƬ��




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  �������˲���������ۣ�  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ S_Sigma_a S_Delta_a] =mtd(s_Sigma_rc,s_Delta_rc,Tr,Fs,Num_Tr_CPI );

S_Sigma_abs=cell(1,3);
S_Delta_abs=cell(1,3);
for m=1:length(Fr)
  S_Sigma_abs{1,m}=abs(S_Sigma_a{1,m});
  S_Delta_abs{1,m}=abs(S_Delta_a{1,m});  
end


figure

s_plot(S_Sigma_abs);
title('��ͨ��MTD���')
xlabel('time [sec]')
ylabel('magnitude [v]')


print(gcf,'-dbitmap','F:\����ͼƬ\MTD���.bmp')   % ����Ϊpng��ʽ��ͼƬ��





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%   CFAR�����龯��⣩  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ S_out] = CFAR(S_Sigma_a,Num_Tr_CPI );

figure
s_plot(S_out);

title('��ͨ��CFAR���')
xlabel('time [sec]')

print(gcf,'-dbitmap','F:\����ͼƬ\CFAR���.bmp')   % ����Ϊpng��ʽ��ͼƬ��



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Ŀ��ȷ�������롢�����ղ���   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s_R s_D Target_R Target_D target_num Target_Range_all Target_Doppler_all ] = measure(S_out,S_Sigma_a,Num_Tr_CPI ,Fs,Tp,Fr,Wavelength);

figure
subplot(2,1,1)
s_plot_B(s_R);
title('�������Ķ�λ����')
xlabel('time [sec]')
ylabel('���� [m]')

subplot(2,1,2)
s_plot_B(s_D);
title('�ٶ����Ķ�λ����')
xlabel('time [sec]')
ylabel('�ٶ� [m/s]')

print(gcf,'-dbitmap','F:\����ͼƬ\����������ն���.bmp')   % ����Ϊpng��ʽ��ͼƬ��



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  �����ģ��   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v_TT=Fr*Wavelength/2;
R_TT=Tr*c/2;
for m=1:length(Fr)
    v_aa(m)=mod(V_set(1),v_TT(m));
    R_aa(m)=mod(R_set(1),R_TT(m));
end


R_am=[Target_R{1,1}(1),Target_R{1,2}(1),Target_R{1,3}(1)];


[ R] = R_ambity(Fr,R_am );
R_plot=[zeros(1,2e3) R zeros(1,500)];

figure
subplot(2,1,1)
plot(R_plot,'LineWidth',2)
title('�����ģ�����')
ylabel('���� [m]')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  ���ٶ�ģ��   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V_am=[Target_D{1,1}(1),Target_D{1,2}(1),Target_D{1,3}(1)];

 [V] =  V_ambity( Fr,V_am );

V_plot=[zeros(1,2e3) V zeros(1,500)];
subplot(2,1,2)
plot(V_plot,'LineWidth',2)
title('���ٶ�ģ�����')
ylabel('�ٶ� [m/s]')

print(gcf,'-dbitmap','F:\����ͼƬ\��ģ�����.bmp')   % ����Ϊpng��ʽ��ͼƬ

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%       ��������       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [angle_aa angle_result ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G);
[angle_result ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G);
[angle_aa ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G);
figure 
s_plot_B(angle_result);
title('�Ͳ�ͨ����ǽ��')
xlabel('time [sec]')
ylabel('�Ƕ� [��]')
print(gcf,'-dbitmap','F:\����ͼƬ\��ǽ��.bmp')   % ����Ϊpng��ʽ��ͼƬ��
figure
angle_s=(angle_aa{1,1}(1)+angle_aa{1,2}(1)+angle_aa{1,3}(1))/3;
angle_plot=[zeros(1,2e3) angle_s zeros(1,500)];
plot(angle_plot,'LineWidth',2)
title('��Ƶ���ȡƽ��')
ylabel('�Ƕ� [��]')

print(gcf,'-dbitmap','F:\����ͼƬ\��Ƶ���ȡƽ��.bmp')   % ����Ϊpng��ʽ��ͼƬ

aaa=1;

