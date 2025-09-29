clear all
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%    雷达系统仿真参数    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=3e8;                           % 光速
k=1.38e-23;                      % 玻尔兹曼常数

Pt=20e3;                         % 发射功率【W】

Fc=1e9;                          % 中心频率【Hz】
Wavelength=c/Fc;                 % 工作波长【m】

Tp=8e-6;                        % 脉冲宽度【微秒】
Fr=[8e3 11e3 13e3];                         % 脉冲重复频率【Hz】

B=10e6;                           % 带宽【Hz】
Fs=20e6;                         % 采样率【Hz】
F=10^(6.99/10);                     % 噪声系数
K=B/Tp;                          % 调频率【Hz】
Tr=1./Fr;% 脉冲重复周期【秒】
R_T=Tr*c/2;%最大模糊距离

Delta_t=1/Fs;                    % 时域采样点时间间隔【秒】
vv=Fr*Wavelength/2;  %最大模糊速度
D=5;                             % 天线孔径【m】
Ae=1*pi*(D/2)^2;                 % 天线有效面积【m^2】
% G=4*pi*Ae/Wavelength^2;          % 天线增益
G=10^(32/10);
BeamWidth=0.88*Wavelength/D;     % 天线3dB波束宽度【deg】
BeamShift=0.8*BeamWidth/2;         % A、B波束与天线轴向的夹角【deg】
Theta0=30*pi/180;                % 波束主瓣初始指向【度】
Wa=0;2*pi/1;                       % 天线波束转速【rad/sec】

Num_Tr_CPI=64;                      % CPI周期数


R_set=[70e3,7e3,10e3];          % 目标距离【m】         
RCS=[1,1,1];                 % 目标平均后向散射截面积【m^2】   
Theta_target_set=30.1*pi/180; % 目标方位角【deg】
V_set=[2100,1000,900];                % 目标速度【m/s】 

for a=1:length(Fr)
    
   R_A(a)=mod(R_set(1),R_T(a));%判断是否出现模糊
end
for a=1:length(Fr)
    
   v_A(a)=mod(V_set(1),vv(a));
end    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      产生发射信号     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s=lfm(Pt,Tp,Fr,B,Fs,G,Num_Tr_CPI);

figure
s_plot(s);
title('雷达发射信号')
xlabel('time [sec]')
ylabel('magnitude [v]')
print(gcf,'-dbitmap','F:\仿真图片\雷达发射信号.bmp')   % 保存为png格式的图片。


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      目标回波     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s_A s_B] = target(G,Fc,Fs,Fr,Num_Tr_CPI,Theta0,Wa,BeamWidth,s,R_set,V_set,RCS,Theta_target_set);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      仿真热噪声     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s_A s_B] = nose(s_A,s_B,k,B,F);

figure
subplot(2,1,1)
s_plot(s_A);
title('A通道回波信号')
xlabel('time [sec]')
ylabel('magnitude [v]')

subplot(2,1,2)
s_plot(s_B);
title('B通道回波信号')
xlabel('time [sec]')
ylabel('magnitude [v]')

print(gcf,'-dbitmap','F:\仿真图片\雷达回波信号.bmp')   % 保存为png格式的图片。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      和差波束调制    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[s_Sigma s_Delta] =sigma_delta(s_A,s_B);


figure
subplot(2,1,1)
s_plot(s_Sigma);
title('和通道回波信号')
xlabel('time [sec]')
ylabel('magnitude [v]')

subplot(2,1,2)
s_plot(s_Delta);
title('差通道回波信号')
xlabel('time [sec]')
ylabel('magnitude [v]')
print(gcf,'-dbitmap','F:\仿真图片\和差调制回波信号.bmp')   % 保存为png格式的图片。



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  匹配滤波（脉冲压缩)  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%]
[s_Sigma_rc s_Delta_rc] = match(s_Sigma,s_Delta,Tr,Fs,K,Num_Tr_CPI);

figure
s_plot(s_Sigma_rc);
title('和通道匹配滤波结果')
xlabel('time [sec]')
ylabel('magnitude [v]')


print(gcf,'-dbitmap','F:\仿真图片\匹配滤波结果.bmp')   % 保存为png格式的图片。




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  多普勒滤波（脉冲积累）  %%%%%
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
title('和通道MTD结果')
xlabel('time [sec]')
ylabel('magnitude [v]')


print(gcf,'-dbitmap','F:\仿真图片\MTD结果.bmp')   % 保存为png格式的图片。





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%   CFAR（恒虚警检测）  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ S_out] = CFAR(S_Sigma_a,Num_Tr_CPI );

figure
s_plot(S_out);

title('和通道CFAR结果')
xlabel('time [sec]')

print(gcf,'-dbitmap','F:\仿真图片\CFAR结果.bmp')   % 保存为png格式的图片。



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  目标确定，距离、多普勒测量   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s_R s_D Target_R Target_D target_num Target_Range_all Target_Doppler_all ] = measure(S_out,S_Sigma_a,Num_Tr_CPI ,Fs,Tp,Fr,Wavelength);

figure
subplot(2,1,1)
s_plot_B(s_R);
title('距离中心定位后结果')
xlabel('time [sec]')
ylabel('距离 [m]')

subplot(2,1,2)
s_plot_B(s_D);
title('速度中心定位后结果')
xlabel('time [sec]')
ylabel('速度 [m/s]')

print(gcf,'-dbitmap','F:\仿真图片\距离与多普勒定心.bmp')   % 保存为png格式的图片。



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  解距离模糊   %%%%%
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
title('解距离模糊结果')
ylabel('距离 [m]')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  解速度模糊   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V_am=[Target_D{1,1}(1),Target_D{1,2}(1),Target_D{1,3}(1)];

 [V] =  V_ambity( Fr,V_am );

V_plot=[zeros(1,2e3) V zeros(1,500)];
subplot(2,1,2)
plot(V_plot,'LineWidth',2)
title('解速度模糊结果')
ylabel('速度 [m/s]')

print(gcf,'-dbitmap','F:\仿真图片\解模糊结果.bmp')   % 保存为png格式的图片

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%       单脉冲测角       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [angle_aa angle_result ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G);
[angle_result ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G);
[angle_aa ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G);
figure 
s_plot_B(angle_result);
title('和差通道测角结果')
xlabel('time [sec]')
ylabel('角度 [°]')
print(gcf,'-dbitmap','F:\仿真图片\测角结果.bmp')   % 保存为png格式的图片。
figure
angle_s=(angle_aa{1,1}(1)+angle_aa{1,2}(1)+angle_aa{1,3}(1))/3;
angle_plot=[zeros(1,2e3) angle_s zeros(1,500)];
plot(angle_plot,'LineWidth',2)
title('重频测角取平均')
ylabel('角度 [°]')

print(gcf,'-dbitmap','F:\仿真图片\重频测角取平均.bmp')   % 保存为png格式的图片

aaa=1;

