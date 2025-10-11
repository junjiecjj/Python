% 某末制导雷达系统设计
% 参数定义
clc;clear all;close all;
FFTsize=4096;%FFT点数
d=0.65;%天线间隔0.65m
C=3e8;%光速
lamda=0.03;%波长
TimeWidth=160e-6;%脉冲宽度
BandWidth=1e6;%调频带宽
K=BandWidth/TimeWidth;%调频斜率
D=0.25;%天线孔径
Ae=0.25*pi*D^2;%天线有效面积
G=4*pi*Ae/lamda^2;%天线增益
RCS=1500;%目标RCS
k=1.38e-23;%玻尔兹曼常数
T=290;%标准噪声温度
F=3;%噪声系数(dB)
L=4;%系统损耗(dB)
Lp=5;%信号处理损失(dB)
N_CI=64;%相干脉冲积累数
Pt_CI=30;%64脉冲相干积累时发射功率
Ru=80000;%不模糊探测距离（m）
theta_3dB=6;%天线波束宽度(角度)
PRT=800e-6;%搜索状态脉冲重复周期
Fs=2e6;%采样频率
Ts=1/Fs;%采样周期
Va=600;%导弹速度
Vs=20;%目标航速
alpha=30;%目标航向与弹轴方向夹角(角度)
beta=1;%目标偏离弹轴方向夹角(角度)
Rs=20000;%第七题中目标距离
nTr=fix(PRT*Fs);%每个脉冲重复周期采样点
nTe=fix(TimeWidth*Fs);%匹配滤波点数
nTe=nTe+mod(nTe,2);
P_fa=10e-6;%虚警概率

% (1)模糊函数和-4dB等高线图
eps=1e-10;%防止奇异点出现
tau=-TimeWidth:TimeWidth/1600:TimeWidth-TimeWidth/1600;
fd=-BandWidth:BandWidth/1000:BandWidth-BandWidth/1000;
[X,Y]=meshgrid(tau,fd);
temp1=1-abs(X)./TimeWidth;
temp2=pi*TimeWidth*(K*X+Y).*temp1+eps;
ambg=abs(temp1.*sin(temp2)./temp2);%模糊函数模值
figure;mesh(tau*1e6,fd*1e-6,ambg);%模糊函数图
xlabel('\tau/us');ylabel('fd/MHz');title('模糊函数图');grid on;
[m1,m2]=find(ambg==max(max(ambg)));%找到原点
figure;plot(tau*1e6,20*log10(abs(ambg(m1,:))));%画出距离模糊函数图
xlabel('\tau/us');ylabel('|X(\tau,0)|');title('|X(\tau,0)|距离模糊图');grid on;axis([-100,100,-60,0]);
figure;plot((fd)*1e-6,20*log10(abs(ambg(:,m2))));%画出速度模糊函数图
xlabel('fd/MHz');ylabel('|X(0,fd)|');title('|X(0,fd)|速度模糊图');grid on;axis([-1,1,-60,0]);
figure;contour(tau*1e6,fd*1e-6,ambg,'bl');%画出模糊函数的等高线图
xlabel('\tau/us');ylabel('fd/MHz');title('模糊函数的等高线图');axis([-150,150,-1,1]);
figure;contour(tau*1e6,fd*1e-6,ambg,[10^(-4/20),10^(-4/20)],'bl');%画出-4dB等高线图
xlabel('\tau/us');ylabel('fd/MHz');title('模糊函数的-4dB切割等高线图局部放大');axis([-2,2,-0.01,0.01]);


[I2,J2]=find(abs(20*log10(ambg)-(-3))<0.1);%计算-3dB时宽带宽
tau_3db=abs(tau(J2(end))-tau(J2(1)))*1e6
B_3db=abs(fd(I2(end))-fd(I2(1)))*1e-6

% （2）（3）略

% （4）相干积累提升SNR
N_pulse=theta_3dB/60/PRT
R_max=100000;%测试最大距离100km
R_CI=linspace(0+R_max/400,R_max,400);
SNR_1=10*log10(Pt_CI*TimeWidth*G^2*RCS*lamda^2)-10*log10((4*pi)^3*k*T.*(R_CI).^4)-F-L-Lp;
SNR_N=SNR_1+10*log10(N_CI);
figure;plot(R_CI*1e-3,SNR_1,'b-.',R_CI*1e-3,SNR_N,'r-');title('相干积累前后信噪比-距离关系曲线');
xlabel('R/km');ylabel('SNR/dB');legend('相干积累前','相干积累后');grid on;axis([30,100,-10,40]);

%%% （5）脉压
t=(-nTe/2:(nTe/2-1))/nTe*TimeWidth;
f=(-256:255)/512*(2*BandWidth);
Slfm=exp(1j*pi*K*t.*t);%线性调频信号
% figure;plot(t*1e6,real(Slfm),'r-',t*1e6,imag(Slfm),'b-.');title('线性调频信号');
Ht=conj(fliplr(Slfm));%时域匹配滤波函数
Hf=fftshift(fft(Ht,512));
figure;plot(t*1e6,real(Ht),'r-',t*1e6,imag(Ht),'b-.');title('线性调频信号匹配滤波函数h(t)');
xlabel('时延/us');ylabel('幅度');legend('实部','虚部');grid on;
figure;plot(f*1e-6,abs(Hf));title('线性调频信号匹配滤波器H(f)');
xlabel('频率/MHz');ylabel('幅度');grid on;
Echo=zeros(1,fix(PRT*Fs));DelayNumber=fix(2*Ru/C*Fs);%目标距离80km
Echo(1,(DelayNumber+1):(DelayNumber+length(Slfm)))=Slfm;%产生回波信号
% figure;plot((0:1/Fs:PRT-1/Fs)*C/2*1e6,real(Echo),'r-',(0:1/Fs:PRT-1/Fs)*C/2*1e6,imag(Echo),'b-.');
% title('回波信号');xlabel('距离/Km');ylabel('幅度');legend('实部','虚部');grid on;
Echo_fft=fft(Echo,2048);%要在频域脉压
Echo_s=repmat(Echo_fft,2,1).';%因为fft按列计算，因此数据按列放置
Ht_s=repmat(Ht,2,1).';%因为fft按列计算，因此数据按列放置
window=[ones(nTe,1),taylorwin(nTe,5,-35)];%窗函数，泰勒窗-35dB
Hf_s=fft(Ht_s.*window,2048);%脉压和加窗的频域
Echo_temp=ifft(Echo_s.*Hf_s);%脉压后未去暂态点
Echo_pc=Echo_temp(nTe:nTe+fix(PRT*Fs)-1,:);%去掉暂态点
% figure;plot((0:fix(PRT*Fs)-1)*C/2/Fs*1e-3,20*log10(abs(Echo_pc(:,:))));%按列画出
% xlabel('距离/Km');ylabel('幅度/dB');grid on;title('回波信号脉压结果');legend('不加窗','加三角窗' ,'加汉明窗');
PC_max=max(20*log10(abs(Echo_pc)));%找出不同窗时的最大值
PC_lose=PC_max-PC_max(1)%显示出加窗后的最大值损失
Slfm_pc=20*log10(abs(Echo_pc((DelayNumber-nTe/2+1):(DelayNumber+nTe/2),:)));%线性调频信号的脉压结果
figure;plot(Slfm_pc(:,1)-PC_max(1));hold on;%归一化后结果 
plot(Slfm_pc(:,2)-PC_max(1),'r');hold on;%归一化后结果 
xlabel('时延/us');ylabel('幅度/dB');grid on;title('回波信号归一化脉压结果');legend('不加窗','加泰勒窗');
axis([0,320,-60,0]);

 % （6）搜索状态仿真
 V=Vs*cos((alpha+beta)/180*pi)%目标与导弹相对速度
Signal_ad=2^8*(Echo/max(abs(Echo)));%信号经过ad后
t_N=0:1/Fs:N_CI*PRT-1/Fs;
Signal_N=repmat(Signal_ad,1,N_CI);%64个周期回波(无噪声)
Signal_N=Signal_N.*exp(1j*2*pi*(2*V/lamda)*t_N);%引入多普勒频移
Noise_N=1/sqrt(2)*(normrnd(0,2^10,1,N_CI*nTr)+1j*normrnd(0,2^10,1,N_CI*nTr));% 噪声信号
Echo_N=Signal_N+Noise_N;%加噪声后的回波信号
Echo_N=reshape(Echo_N,nTr,N_CI);
figure;plot(abs(Echo_N));title('原始信号');%回波基带信号图像
xlabel('时域采样点');ylabel('幅度(模值)');grid on;
Echo_N_fft=fft(Echo_N,2048);%回波信号FFT
Hf_N=fft(Ht,2048);%频域脉压系数
Hf_N=repmat(Hf_N.',1,N_CI);%脉压系数矩阵
Echo_N_temp=ifft(Echo_N_fft.*Hf_N);%频域脉压，未去暂态点
Echo_N_pc=Echo_N_temp(nTe:nTe+nTr-1,:);%去掉暂态点
figure;plot((0:nTr-1)/Fs*C/2*1e-3,20*log10(abs(Echo_N_pc.')));title('回波信号脉压结果');
xlabel('距离单元/Km');ylabel('幅度/dB');grid on;
Echo_N_mtd=fftshift(fft(Echo_N_pc.'),1);%64脉冲相干积累和MTD
figure;mesh((0:nTr-1)/Fs*C/2*1e-3,(-32:31)/PRT/64,abs(Echo_N_mtd));
xlabel('距离/Km');ylabel('多谱勒频率/Hz');zlabel('幅度');grid on;title('64脉冲相干积累结果');set(gcf,'color',[1,1,1])
figure;contour((0:nTr-1)/Fs*C/2*1e-3,(-32:31)/PRT/64,abs(Echo_N_mtd),'bl');%画出相干积累的等高线图
xlabel('距离/Km');ylabel('多谱勒频率/Hz');zlabel('幅度');grid on;title('64脉冲相干积累等高线图');
[index_i index_j]=find(abs(Echo_N_mtd)==max(max(abs(Echo_N_mtd))));%找最大值多普勒单元对应的重复周期进行CFAR处理
V_fd=2*V/lamda %多普勒素的对应的多普勒频率
mtd_fd=(index_i-1)/PRT/64% 相参积累对应的多普勒频率
SNR_echo=20*log10(2^8/2^10)%原始回波基带信号信噪比
SNR_pc=SNR_echo+10*log10(BandWidth*TimeWidth)%脉压后信噪比
SNR_ci=SNR_pc+10*log10(64)%64脉冲相干积累后信噪比

%恒虚警
N_mean=8;N_baohu=4;K0_CFAR=(1/P_fa)^(1/N_mean)-1%计算系数K
CFAR_data=abs(Echo_N_mtd(index_i,:));%信号
K_CFAR=K0_CFAR./N_mean.*[ones(1,N_mean/2) zeros(1,N_baohu+1) ones(1,N_mean/2)];%恒虚警系数向量
CFAR_noise=conv(CFAR_data,K_CFAR);%恒虚警处理
CFAR_noise=CFAR_noise(length(K_CFAR):length(CFAR_data));%去暂态点
figure;plot(((N_mean+N_baohu)/2:nTr-(N_mean+N_baohu)/2-1)/Fs*C/2*1e-3,20*log10(CFAR_noise),'r-.');%恒虚警电平
hold on;plot((0:nTr-1)/Fs*C/2*1e-3,20*log10(CFAR_data),'b-');%信号
xlabel('距离/Km');ylabel('幅度');grid on;title('恒虚警处理');legend('恒虚警电平','信号电平');hold off

% (7)单脉冲测角
theta=(-theta_3dB:theta_3dB/1200:theta_3dB-theta_3dB/1200)*pi/180;%波束宽度范围内
Ftheta1=exp(-2.778*((theta-theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180).^2);%波束1方向图函数
Ftheta2=exp(-2.778*((theta+theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180).^2);%波束2方向图函数
Fsum=Ftheta1+Ftheta2;%和波束
Fdif=Ftheta1-Ftheta2;%差波束
Ferr=real(Fsum.*conj(Fdif))./(Fsum.*conj(Fsum));%归一化误差信号
figure;plot(theta*180/pi,Ftheta1,'r-.',theta*180/pi,Ftheta2,'b-.');
hold on;plot(theta*180/pi,Fsum,'r',theta*180/pi,Fdif,'b');
xlabel('角度/度');ylabel('幅度');grid on;title('和差波束');hold on;
plot(theta*180/pi,Ferr,'k');legend('波束1','波束2','和波束','差波束','误差信号');
K_theta=polyfit(theta(1100:1300)*180/pi,Ferr(1100:1300),1);
fprintf('误差信号斜率为 %5.4f\n',1/K_theta(1));

% 计算偏离电轴中心0.5°，1°时的归一化误差信号
theta_pianli=[0.5 1];
Ferr_pinli=theta_pianli*K_theta(1);
fprintf('0.5°，1°的误差信号为 %5.4f,%5.4f \n',Ferr_pinli);

% Monto Carlo分析
SNR_MC=5:30;N_MC=100;%Monto Carlo次数
Ftheta1_MC=exp(-2.778*((0-theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180)^2);%无偏时波束1方向图函数
Ftheta2_MC=exp(-2.778*((0+theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180)^2);%无偏时波束2方向图函数
Fsum_MC=repmat(Ftheta1_MC+Ftheta2_MC,N_MC,length(SNR_MC));%无偏时和波束
Fdif_MC=repmat(Ftheta1_MC-Ftheta2_MC,N_MC,length(SNR_MC));%无偏时差波束
Nsum_MC=1/sqrt(2)*(normrnd(0,1,N_MC,1)+1j*normrnd(0,1,N_MC,1))*10.^(-SNR_MC/20);%和通道噪声信号
Ndif_MC=1/sqrt(2)*(normrnd(0,1,N_MC,1)+1j*normrnd(0,1,N_MC,1))*10.^(-SNR_MC/20);%差通道噪声信号
Echo_sum_MC=Fsum_MC+Nsum_MC;Echo_dif_MC=Fdif_MC+Ndif_MC;%和差通道加入噪声
theta_MC=real(Echo_sum_MC.*conj(Echo_dif_MC))./(Echo_sum_MC.*conj(Echo_sum_MC));%归一化误差信号
theta_guji=theta_MC./K_theta(1);%角度估计
figure;plot(SNR_MC,theta_guji','.');grid on;
xlabel('SNR/dB');ylabel('单词测量误差/度');title('SNR与单次测角误差');
std_theta_wucha=std(theta_guji);%测角均方根误差
figure;plot(SNR_MC,std_theta_wucha);grid on;title('均方根误差');
xlabel('SNR/dB');ylabel('均方根误差/°');

%和差通道时域脉压
theta_hecha=1;%方位角/度
SNR=20;%信噪比20dB
Ftheta1_hecha=exp(-2.778*((theta_hecha*pi/180-theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180)^2);%偏角为1度时波束1方向图函数
Ftheta2_hecha=exp(-2.778*((theta_hecha*pi/180+theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180)^2);%偏角为1度时波束2方向图函数
Signal_hecha=zeros(1,fix(PRT*Fs));DelayNumber_hecha=fix(2*Rs/C*Fs);
Signal_hecha(1,(DelayNumber_hecha+1):(DelayNumber_hecha+length(Slfm)))=Slfm;%产生回波信号(无噪声)
Signal_hecha=(Signal_hecha/max(abs(Signal_hecha)));%信号功率归一化
Echo_boshu1=Signal_hecha+1/sqrt(2)*(normrnd(0,1,1,nTr)+1j*normrnd(0,1,1,nTr))*10^(-SNR/20);%波束1回波
Echo_boshu2=Signal_hecha+1/sqrt(2)*(normrnd(0,1,1,nTr)+1j*normrnd(0,1,1,nTr))*10^(-SNR/20);%波束1回波
Echo_sum_hecha=Ftheta1_hecha*Echo_boshu1+Ftheta2_hecha*Echo_boshu2;%偏角为1度时和波束
Echo_dif_hecha=Ftheta1_hecha*Echo_boshu1-Ftheta2_hecha*Echo_boshu2;%偏角为1度时差波束
Echo_sum_pc=conv(Echo_sum_hecha,Ht);Echo_sum_pc=Echo_sum_pc(nTe:nTe+nTr-1);%和通道时域脉压
Echo_dif_pc=conv(Echo_dif_hecha,Ht);Echo_dif_pc=Echo_dif_pc(nTe:nTe+nTr-1);%差通道时域脉压
Echo_err=real(Echo_sum_pc.*conj(Echo_dif_pc))./(Echo_sum_pc.*conj(Echo_sum_pc));%归一化误差
figure;plot((0:1/Fs:PRT-1/Fs)*C/2*1e-3,20*log10(abs(Echo_sum_pc)));title('弹目距离20Km时和通道时域脉压结果');
xlabel('距离/Km');ylabel('幅度');grid on;
figure;plot((0:1/Fs:PRT-1/Fs)*C/2*1e-3,20*log10(abs(Echo_dif_pc)));title('弹目距离20Km时差通道时域脉压结果');
xlabel('距离/Km');ylabel('幅度');grid on;
fprintf('设定20km处方位角%5.4f°，误差信号为 %5.4f ，测量值为 %5.4f°\n',theta_hecha,Echo_err(266),Echo_err(266)*(1/K_theta(1)));
% figure;plot((0:1/Fs:PRT-1/Fs)*C/2*1e-3,20*log10(Echo_err));title('弹目距离20Km时归一化误差');
% xlabel('距离/Km');ylabel('归一化误差');grid on;

%% (8)中频正交采样
F_if=60e6;%接收机中频
M_ad=3;Fs_ad=4*F_if/(2*M_ad-1)%%ad采样频率
BandWidth_track=10e6;%跟踪时调频带宽
TimeWidth_track=10e-6;%跟踪时调频时度
nTe_track=fix(TimeWidth_track*Fs_ad);%匹配滤波点数
nTe_track=nTe_track+mod(nTe_track,2);
t_track=(-nTe_track/2:(nTe_track/2-1))/nTe_track*TimeWidth_track;
Slfm_track=cos(2*pi*(F_if*t_track+BandWidth_track/TimeWidth_track/2*t_track.*t_track));%跟踪时线性调频信号
Modify=[1,-1,-1,1;1,1,-1,-1];%符号修正矩阵
Slfm_track=Slfm_track.*kron(ones(1,nTe_track/4),Modify(mod(M_ad,2)+1,:));%符号修正

%低通滤波器设计
f_lowpass= [BandWidth_track,1.2*BandWidth_track];%通带截止频率Bs,阻带截止频率1.2Bs 
a_lowpass= [1,0];
Rp_lowpass=1;Rs_lowpass=40;%通带最大衰减1dB，阻带最小衰减40dB
dev_lowpass= [(10^(Rp_lowpass/20)-1)/(10^(Rp_lowpass/20)+1),10^(-Rs_lowpass/20)];%通带和阻带震荡波纹幅度
[n_lowpass,fo_lowpass,ao_lowpass,w_lowpass] = firpmord(f_lowpass,a_lowpass,dev_lowpass,Fs_ad);%滤波器参数
n_lowpass=n_lowpass+1-mod(n_lowpass,2)
h_lowpass= firpm(n_lowpass,fo_lowpass,ao_lowpass,w_lowpass);
figure;freqz(h_lowpass,1,1024,Fs_ad);title('低通滤波器响应');

%iq分解
I_track=filter(h_lowpass(2:2:end),1,Slfm_track(1:2:end));
Q_track=filter(h_lowpass(1:2:end),1,Slfm_track(2:2:end));
Sig_track=I_track(n_lowpass/2:end)+1j*Q_track(n_lowpass/2:end);
figure;plot(Sig_track,'.');axis([-0.6 0.6 -0.6 0.6]);axis square;
grid on;xlabel('I通道');ylabel('Q通道');title('IQ正交性');

%产生中频正交采样信号
R_target=10000;%设定目标距离10km
SNR_IQ=20;%设定信噪比5dB
% figure;subplot(221);plot(20*log10(abs(Slfm_track(1:2:end))));
% subplot(222);plot(20*log10(abs(Slfm_track(2:2:end))));
% Slfm_complex=Slfm_track(1:2:end)+j*Slfm_track(2:2:end);%变为复数形式
Slfmexp_track=exp(j*2*pi*(F_if*t_track+BandWidth_track/TimeWidth_track/2*t_track.*t_track));%跟踪时线性调频信号
Ht_track=conj(fliplr(Slfmexp_track));%时域匹配滤波函数
Echo=zeros(1,fix(PRT*Fs_ad));DelayNumber=fix(2*R_target/C*Fs_ad);%目标距离12km对应的延迟
Echo(1,(DelayNumber+1):(DelayNumber+length(Slfm_track)))=Slfm_track;%产生回波信号
% Echo_noise=1/sqrt(2)*normrnd(0,1,1,size(Echo,2))*10.^(-SNR_IQ/20);%通道噪声信号
% Echo=Echo+Echo_noise;%通道加入噪声
figure;plot(20*log10(abs(Echo)));grid on;axis([3200,3700,-40,0]);
xlabel('距离单元');ylabel('幅度/dB');title('目标回波中频信号');
% figure;plot(abs(Echo));grid on;
% xlabel('距离单元');ylabel('幅度');title('目标回波中频信号');
cos_I=cos(2*pi*F_if/Fs_ad.*[1:length(Echo)]);
sin_Q=sin(2*pi*F_if/Fs_ad.*[1:length(Echo)]);
Echo_I=Echo.*cos_I;
Echo_Q=-Echo.*sin_Q;
I_track=filter(h_lowpass(2:2:end),1,Echo_I(1:end));
Q_track=filter(h_lowpass(1:2:end),1,Echo_Q(1:end));%产生IQ基带信号
figure;subplot(211);plot(20*log10(abs(I_track)));axis([3200,3700,-40,0]);
xlabel('距离单元');ylabel('幅度/dB');title('正交采样I通道基带信号');grid on;
subplot(212);plot(20*log10(abs(Q_track)));axis([3200,3700,-40,0]);
xlabel('距离单元');ylabel('幅度/dB');title('正交采样Q通道基带信号');grid on;

% figure;subplot(211);plot((abs(I_track)));axis([3200,3700,0,0.5]);
% xlabel('距离单元');ylabel('幅度');title('正交采样I通道基带信号');grid on;
% subplot(212);plot((abs(Q_track)));axis([3200,3700,0,0.5]);
% xlabel('距离单元');ylabel('幅度');title('正交采样Q通道基带信号');grid on;
Echo_IQ_track=I_track(n_lowpass/2:end)+1j.*Q_track(n_lowpass/2:end);
figure;plot(Echo_IQ_track,'.');axis square;grid on;
xlabel('I通道');ylabel('Q通道');title('IQ正交性');
Echo_IQ_track_pc=conv(Echo_IQ_track,Ht_track);
Echo_IQ_track_pc=Echo_IQ_track_pc(length(Ht_track):end);%对正交采样的基带信号进行脉压处理
figure;plot((0:1/(Fs_ad):PRT-((1+fix(n_lowpass/2))/Fs_ad))*C/2*1e-3,20*log10(abs(Echo_IQ_track_pc)));
xlabel('距离/km');ylabel('幅度/dB');title('基带信号脉压处理');axis([0,80,-20,50]);grid on;axis([8,12,-10,50]);
%  figure;plot((0:1/(Fs_ad):PRT-((1+fix(n_lowpass/2))/Fs_ad))*C/2*1e-3,abs(Echo_IQ_track_pc));
%  xlabel('距离/km');ylabel('幅度');title('基带信号脉压处理');grid on;

% figure;plot(20*log10(abs(Echo_IQ_track_pc)));
% xlabel('距离单元');ylabel('幅度/dB');title('基带信号脉压处理');grid on;
%脉压后目标延迟变为1920，而不是1929，而且在1929和1930上都有尖峰，是滤波器设计的原因？


