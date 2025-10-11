% ĳĩ�Ƶ��״�ϵͳ���
% ��������
clc;clear all;close all;
FFTsize=4096;%FFT����
d=0.65;%���߼��0.65m
C=3e8;%����
lamda=0.03;%����
TimeWidth=160e-6;%������
BandWidth=1e6;%��Ƶ����
K=BandWidth/TimeWidth;%��Ƶб��
D=0.25;%���߿׾�
Ae=0.25*pi*D^2;%������Ч���
G=4*pi*Ae/lamda^2;%��������
RCS=1500;%Ŀ��RCS
k=1.38e-23;%������������
T=290;%��׼�����¶�
F=3;%����ϵ��(dB)
L=4;%ϵͳ���(dB)
Lp=5;%�źŴ�����ʧ(dB)
N_CI=64;%������������
Pt_CI=30;%64������ɻ���ʱ���书��
Ru=80000;%��ģ��̽����루m��
theta_3dB=6;%���߲������(�Ƕ�)
PRT=800e-6;%����״̬�����ظ�����
Fs=2e6;%����Ƶ��
Ts=1/Fs;%��������
Va=600;%�����ٶ�
Vs=20;%Ŀ�꺽��
alpha=30;%Ŀ�꺽���뵯�᷽��н�(�Ƕ�)
beta=1;%Ŀ��ƫ�뵯�᷽��н�(�Ƕ�)
Rs=20000;%��������Ŀ�����
nTr=fix(PRT*Fs);%ÿ�������ظ����ڲ�����
nTe=fix(TimeWidth*Fs);%ƥ���˲�����
nTe=nTe+mod(nTe,2);
P_fa=10e-6;%�龯����

% (1)ģ��������-4dB�ȸ���ͼ
eps=1e-10;%��ֹ��������
tau=-TimeWidth:TimeWidth/1600:TimeWidth-TimeWidth/1600;
fd=-BandWidth:BandWidth/1000:BandWidth-BandWidth/1000;
[X,Y]=meshgrid(tau,fd);
temp1=1-abs(X)./TimeWidth;
temp2=pi*TimeWidth*(K*X+Y).*temp1+eps;
ambg=abs(temp1.*sin(temp2)./temp2);%ģ������ģֵ
figure;mesh(tau*1e6,fd*1e-6,ambg);%ģ������ͼ
xlabel('\tau/us');ylabel('fd/MHz');title('ģ������ͼ');grid on;
[m1,m2]=find(ambg==max(max(ambg)));%�ҵ�ԭ��
figure;plot(tau*1e6,20*log10(abs(ambg(m1,:))));%��������ģ������ͼ
xlabel('\tau/us');ylabel('|X(\tau,0)|');title('|X(\tau,0)|����ģ��ͼ');grid on;axis([-100,100,-60,0]);
figure;plot((fd)*1e-6,20*log10(abs(ambg(:,m2))));%�����ٶ�ģ������ͼ
xlabel('fd/MHz');ylabel('|X(0,fd)|');title('|X(0,fd)|�ٶ�ģ��ͼ');grid on;axis([-1,1,-60,0]);
figure;contour(tau*1e6,fd*1e-6,ambg,'bl');%����ģ�������ĵȸ���ͼ
xlabel('\tau/us');ylabel('fd/MHz');title('ģ�������ĵȸ���ͼ');axis([-150,150,-1,1]);
figure;contour(tau*1e6,fd*1e-6,ambg,[10^(-4/20),10^(-4/20)],'bl');%����-4dB�ȸ���ͼ
xlabel('\tau/us');ylabel('fd/MHz');title('ģ��������-4dB�и�ȸ���ͼ�ֲ��Ŵ�');axis([-2,2,-0.01,0.01]);


[I2,J2]=find(abs(20*log10(ambg)-(-3))<0.1);%����-3dBʱ�����
tau_3db=abs(tau(J2(end))-tau(J2(1)))*1e6
B_3db=abs(fd(I2(end))-fd(I2(1)))*1e-6

% ��2����3����

% ��4����ɻ�������SNR
N_pulse=theta_3dB/60/PRT
R_max=100000;%����������100km
R_CI=linspace(0+R_max/400,R_max,400);
SNR_1=10*log10(Pt_CI*TimeWidth*G^2*RCS*lamda^2)-10*log10((4*pi)^3*k*T.*(R_CI).^4)-F-L-Lp;
SNR_N=SNR_1+10*log10(N_CI);
figure;plot(R_CI*1e-3,SNR_1,'b-.',R_CI*1e-3,SNR_N,'r-');title('��ɻ���ǰ�������-�����ϵ����');
xlabel('R/km');ylabel('SNR/dB');legend('��ɻ���ǰ','��ɻ��ۺ�');grid on;axis([30,100,-10,40]);

%%% ��5����ѹ
t=(-nTe/2:(nTe/2-1))/nTe*TimeWidth;
f=(-256:255)/512*(2*BandWidth);
Slfm=exp(1j*pi*K*t.*t);%���Ե�Ƶ�ź�
% figure;plot(t*1e6,real(Slfm),'r-',t*1e6,imag(Slfm),'b-.');title('���Ե�Ƶ�ź�');
Ht=conj(fliplr(Slfm));%ʱ��ƥ���˲�����
Hf=fftshift(fft(Ht,512));
figure;plot(t*1e6,real(Ht),'r-',t*1e6,imag(Ht),'b-.');title('���Ե�Ƶ�ź�ƥ���˲�����h(t)');
xlabel('ʱ��/us');ylabel('����');legend('ʵ��','�鲿');grid on;
figure;plot(f*1e-6,abs(Hf));title('���Ե�Ƶ�ź�ƥ���˲���H(f)');
xlabel('Ƶ��/MHz');ylabel('����');grid on;
Echo=zeros(1,fix(PRT*Fs));DelayNumber=fix(2*Ru/C*Fs);%Ŀ�����80km
Echo(1,(DelayNumber+1):(DelayNumber+length(Slfm)))=Slfm;%�����ز��ź�
% figure;plot((0:1/Fs:PRT-1/Fs)*C/2*1e6,real(Echo),'r-',(0:1/Fs:PRT-1/Fs)*C/2*1e6,imag(Echo),'b-.');
% title('�ز��ź�');xlabel('����/Km');ylabel('����');legend('ʵ��','�鲿');grid on;
Echo_fft=fft(Echo,2048);%Ҫ��Ƶ����ѹ
Echo_s=repmat(Echo_fft,2,1).';%��Ϊfft���м��㣬������ݰ��з���
Ht_s=repmat(Ht,2,1).';%��Ϊfft���м��㣬������ݰ��з���
window=[ones(nTe,1),taylorwin(nTe,5,-35)];%��������̩�մ�-35dB
Hf_s=fft(Ht_s.*window,2048);%��ѹ�ͼӴ���Ƶ��
Echo_temp=ifft(Echo_s.*Hf_s);%��ѹ��δȥ��̬��
Echo_pc=Echo_temp(nTe:nTe+fix(PRT*Fs)-1,:);%ȥ����̬��
% figure;plot((0:fix(PRT*Fs)-1)*C/2/Fs*1e-3,20*log10(abs(Echo_pc(:,:))));%���л���
% xlabel('����/Km');ylabel('����/dB');grid on;title('�ز��ź���ѹ���');legend('���Ӵ�','�����Ǵ�' ,'�Ӻ�����');
PC_max=max(20*log10(abs(Echo_pc)));%�ҳ���ͬ��ʱ�����ֵ
PC_lose=PC_max-PC_max(1)%��ʾ���Ӵ�������ֵ��ʧ
Slfm_pc=20*log10(abs(Echo_pc((DelayNumber-nTe/2+1):(DelayNumber+nTe/2),:)));%���Ե�Ƶ�źŵ���ѹ���
figure;plot(Slfm_pc(:,1)-PC_max(1));hold on;%��һ������ 
plot(Slfm_pc(:,2)-PC_max(1),'r');hold on;%��һ������ 
xlabel('ʱ��/us');ylabel('����/dB');grid on;title('�ز��źŹ�һ����ѹ���');legend('���Ӵ�','��̩�մ�');
axis([0,320,-60,0]);

 % ��6������״̬����
 V=Vs*cos((alpha+beta)/180*pi)%Ŀ���뵼������ٶ�
Signal_ad=2^8*(Echo/max(abs(Echo)));%�źž���ad��
t_N=0:1/Fs:N_CI*PRT-1/Fs;
Signal_N=repmat(Signal_ad,1,N_CI);%64�����ڻز�(������)
Signal_N=Signal_N.*exp(1j*2*pi*(2*V/lamda)*t_N);%���������Ƶ��
Noise_N=1/sqrt(2)*(normrnd(0,2^10,1,N_CI*nTr)+1j*normrnd(0,2^10,1,N_CI*nTr));% �����ź�
Echo_N=Signal_N+Noise_N;%��������Ļز��ź�
Echo_N=reshape(Echo_N,nTr,N_CI);
figure;plot(abs(Echo_N));title('ԭʼ�ź�');%�ز������ź�ͼ��
xlabel('ʱ�������');ylabel('����(ģֵ)');grid on;
Echo_N_fft=fft(Echo_N,2048);%�ز��ź�FFT
Hf_N=fft(Ht,2048);%Ƶ����ѹϵ��
Hf_N=repmat(Hf_N.',1,N_CI);%��ѹϵ������
Echo_N_temp=ifft(Echo_N_fft.*Hf_N);%Ƶ����ѹ��δȥ��̬��
Echo_N_pc=Echo_N_temp(nTe:nTe+nTr-1,:);%ȥ����̬��
figure;plot((0:nTr-1)/Fs*C/2*1e-3,20*log10(abs(Echo_N_pc.')));title('�ز��ź���ѹ���');
xlabel('���뵥Ԫ/Km');ylabel('����/dB');grid on;
Echo_N_mtd=fftshift(fft(Echo_N_pc.'),1);%64������ɻ��ۺ�MTD
figure;mesh((0:nTr-1)/Fs*C/2*1e-3,(-32:31)/PRT/64,abs(Echo_N_mtd));
xlabel('����/Km');ylabel('������Ƶ��/Hz');zlabel('����');grid on;title('64������ɻ��۽��');set(gcf,'color',[1,1,1])
figure;contour((0:nTr-1)/Fs*C/2*1e-3,(-32:31)/PRT/64,abs(Echo_N_mtd),'bl');%������ɻ��۵ĵȸ���ͼ
xlabel('����/Km');ylabel('������Ƶ��/Hz');zlabel('����');grid on;title('64������ɻ��۵ȸ���ͼ');
[index_i index_j]=find(abs(Echo_N_mtd)==max(max(abs(Echo_N_mtd))));%�����ֵ�����յ�Ԫ��Ӧ���ظ����ڽ���CFAR����
V_fd=2*V/lamda %�������صĶ�Ӧ�Ķ�����Ƶ��
mtd_fd=(index_i-1)/PRT/64% ��λ��۶�Ӧ�Ķ�����Ƶ��
SNR_echo=20*log10(2^8/2^10)%ԭʼ�ز������ź������
SNR_pc=SNR_echo+10*log10(BandWidth*TimeWidth)%��ѹ�������
SNR_ci=SNR_pc+10*log10(64)%64������ɻ��ۺ������

%���龯
N_mean=8;N_baohu=4;K0_CFAR=(1/P_fa)^(1/N_mean)-1%����ϵ��K
CFAR_data=abs(Echo_N_mtd(index_i,:));%�ź�
K_CFAR=K0_CFAR./N_mean.*[ones(1,N_mean/2) zeros(1,N_baohu+1) ones(1,N_mean/2)];%���龯ϵ������
CFAR_noise=conv(CFAR_data,K_CFAR);%���龯����
CFAR_noise=CFAR_noise(length(K_CFAR):length(CFAR_data));%ȥ��̬��
figure;plot(((N_mean+N_baohu)/2:nTr-(N_mean+N_baohu)/2-1)/Fs*C/2*1e-3,20*log10(CFAR_noise),'r-.');%���龯��ƽ
hold on;plot((0:nTr-1)/Fs*C/2*1e-3,20*log10(CFAR_data),'b-');%�ź�
xlabel('����/Km');ylabel('����');grid on;title('���龯����');legend('���龯��ƽ','�źŵ�ƽ');hold off

% (7)��������
theta=(-theta_3dB:theta_3dB/1200:theta_3dB-theta_3dB/1200)*pi/180;%������ȷ�Χ��
Ftheta1=exp(-2.778*((theta-theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180).^2);%����1����ͼ����
Ftheta2=exp(-2.778*((theta+theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180).^2);%����2����ͼ����
Fsum=Ftheta1+Ftheta2;%�Ͳ���
Fdif=Ftheta1-Ftheta2;%���
Ferr=real(Fsum.*conj(Fdif))./(Fsum.*conj(Fsum));%��һ������ź�
figure;plot(theta*180/pi,Ftheta1,'r-.',theta*180/pi,Ftheta2,'b-.');
hold on;plot(theta*180/pi,Fsum,'r',theta*180/pi,Fdif,'b');
xlabel('�Ƕ�/��');ylabel('����');grid on;title('�Ͳ��');hold on;
plot(theta*180/pi,Ferr,'k');legend('����1','����2','�Ͳ���','���','����ź�');
K_theta=polyfit(theta(1100:1300)*180/pi,Ferr(1100:1300),1);
fprintf('����ź�б��Ϊ %5.4f\n',1/K_theta(1));

% ����ƫ���������0.5�㣬1��ʱ�Ĺ�һ������ź�
theta_pianli=[0.5 1];
Ferr_pinli=theta_pianli*K_theta(1);
fprintf('0.5�㣬1�������ź�Ϊ %5.4f,%5.4f \n',Ferr_pinli);

% Monto Carlo����
SNR_MC=5:30;N_MC=100;%Monto Carlo����
Ftheta1_MC=exp(-2.778*((0-theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180)^2);%��ƫʱ����1����ͼ����
Ftheta2_MC=exp(-2.778*((0+theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180)^2);%��ƫʱ����2����ͼ����
Fsum_MC=repmat(Ftheta1_MC+Ftheta2_MC,N_MC,length(SNR_MC));%��ƫʱ�Ͳ���
Fdif_MC=repmat(Ftheta1_MC-Ftheta2_MC,N_MC,length(SNR_MC));%��ƫʱ���
Nsum_MC=1/sqrt(2)*(normrnd(0,1,N_MC,1)+1j*normrnd(0,1,N_MC,1))*10.^(-SNR_MC/20);%��ͨ�������ź�
Ndif_MC=1/sqrt(2)*(normrnd(0,1,N_MC,1)+1j*normrnd(0,1,N_MC,1))*10.^(-SNR_MC/20);%��ͨ�������ź�
Echo_sum_MC=Fsum_MC+Nsum_MC;Echo_dif_MC=Fdif_MC+Ndif_MC;%�Ͳ�ͨ����������
theta_MC=real(Echo_sum_MC.*conj(Echo_dif_MC))./(Echo_sum_MC.*conj(Echo_sum_MC));%��һ������ź�
theta_guji=theta_MC./K_theta(1);%�Ƕȹ���
figure;plot(SNR_MC,theta_guji','.');grid on;
xlabel('SNR/dB');ylabel('���ʲ������/��');title('SNR�뵥�β�����');
std_theta_wucha=std(theta_guji);%��Ǿ��������
figure;plot(SNR_MC,std_theta_wucha);grid on;title('���������');
xlabel('SNR/dB');ylabel('���������/��');

%�Ͳ�ͨ��ʱ����ѹ
theta_hecha=1;%��λ��/��
SNR=20;%�����20dB
Ftheta1_hecha=exp(-2.778*((theta_hecha*pi/180-theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180)^2);%ƫ��Ϊ1��ʱ����1����ͼ����
Ftheta2_hecha=exp(-2.778*((theta_hecha*pi/180+theta_3dB/2*pi/180).^2)/(theta_3dB*pi/180)^2);%ƫ��Ϊ1��ʱ����2����ͼ����
Signal_hecha=zeros(1,fix(PRT*Fs));DelayNumber_hecha=fix(2*Rs/C*Fs);
Signal_hecha(1,(DelayNumber_hecha+1):(DelayNumber_hecha+length(Slfm)))=Slfm;%�����ز��ź�(������)
Signal_hecha=(Signal_hecha/max(abs(Signal_hecha)));%�źŹ��ʹ�һ��
Echo_boshu1=Signal_hecha+1/sqrt(2)*(normrnd(0,1,1,nTr)+1j*normrnd(0,1,1,nTr))*10^(-SNR/20);%����1�ز�
Echo_boshu2=Signal_hecha+1/sqrt(2)*(normrnd(0,1,1,nTr)+1j*normrnd(0,1,1,nTr))*10^(-SNR/20);%����1�ز�
Echo_sum_hecha=Ftheta1_hecha*Echo_boshu1+Ftheta2_hecha*Echo_boshu2;%ƫ��Ϊ1��ʱ�Ͳ���
Echo_dif_hecha=Ftheta1_hecha*Echo_boshu1-Ftheta2_hecha*Echo_boshu2;%ƫ��Ϊ1��ʱ���
Echo_sum_pc=conv(Echo_sum_hecha,Ht);Echo_sum_pc=Echo_sum_pc(nTe:nTe+nTr-1);%��ͨ��ʱ����ѹ
Echo_dif_pc=conv(Echo_dif_hecha,Ht);Echo_dif_pc=Echo_dif_pc(nTe:nTe+nTr-1);%��ͨ��ʱ����ѹ
Echo_err=real(Echo_sum_pc.*conj(Echo_dif_pc))./(Echo_sum_pc.*conj(Echo_sum_pc));%��һ�����
figure;plot((0:1/Fs:PRT-1/Fs)*C/2*1e-3,20*log10(abs(Echo_sum_pc)));title('��Ŀ����20Kmʱ��ͨ��ʱ����ѹ���');
xlabel('����/Km');ylabel('����');grid on;
figure;plot((0:1/Fs:PRT-1/Fs)*C/2*1e-3,20*log10(abs(Echo_dif_pc)));title('��Ŀ����20Kmʱ��ͨ��ʱ����ѹ���');
xlabel('����/Km');ylabel('����');grid on;
fprintf('�趨20km����λ��%5.4f�㣬����ź�Ϊ %5.4f ������ֵΪ %5.4f��\n',theta_hecha,Echo_err(266),Echo_err(266)*(1/K_theta(1)));
% figure;plot((0:1/Fs:PRT-1/Fs)*C/2*1e-3,20*log10(Echo_err));title('��Ŀ����20Kmʱ��һ�����');
% xlabel('����/Km');ylabel('��һ�����');grid on;

%% (8)��Ƶ��������
F_if=60e6;%���ջ���Ƶ
M_ad=3;Fs_ad=4*F_if/(2*M_ad-1)%%ad����Ƶ��
BandWidth_track=10e6;%����ʱ��Ƶ����
TimeWidth_track=10e-6;%����ʱ��Ƶʱ��
nTe_track=fix(TimeWidth_track*Fs_ad);%ƥ���˲�����
nTe_track=nTe_track+mod(nTe_track,2);
t_track=(-nTe_track/2:(nTe_track/2-1))/nTe_track*TimeWidth_track;
Slfm_track=cos(2*pi*(F_if*t_track+BandWidth_track/TimeWidth_track/2*t_track.*t_track));%����ʱ���Ե�Ƶ�ź�
Modify=[1,-1,-1,1;1,1,-1,-1];%������������
Slfm_track=Slfm_track.*kron(ones(1,nTe_track/4),Modify(mod(M_ad,2)+1,:));%��������

%��ͨ�˲������
f_lowpass= [BandWidth_track,1.2*BandWidth_track];%ͨ����ֹƵ��Bs,�����ֹƵ��1.2Bs 
a_lowpass= [1,0];
Rp_lowpass=1;Rs_lowpass=40;%ͨ�����˥��1dB�������С˥��40dB
dev_lowpass= [(10^(Rp_lowpass/20)-1)/(10^(Rp_lowpass/20)+1),10^(-Rs_lowpass/20)];%ͨ��������𵴲��Ʒ���
[n_lowpass,fo_lowpass,ao_lowpass,w_lowpass] = firpmord(f_lowpass,a_lowpass,dev_lowpass,Fs_ad);%�˲�������
n_lowpass=n_lowpass+1-mod(n_lowpass,2)
h_lowpass= firpm(n_lowpass,fo_lowpass,ao_lowpass,w_lowpass);
figure;freqz(h_lowpass,1,1024,Fs_ad);title('��ͨ�˲�����Ӧ');

%iq�ֽ�
I_track=filter(h_lowpass(2:2:end),1,Slfm_track(1:2:end));
Q_track=filter(h_lowpass(1:2:end),1,Slfm_track(2:2:end));
Sig_track=I_track(n_lowpass/2:end)+1j*Q_track(n_lowpass/2:end);
figure;plot(Sig_track,'.');axis([-0.6 0.6 -0.6 0.6]);axis square;
grid on;xlabel('Iͨ��');ylabel('Qͨ��');title('IQ������');

%������Ƶ���������ź�
R_target=10000;%�趨Ŀ�����10km
SNR_IQ=20;%�趨�����5dB
% figure;subplot(221);plot(20*log10(abs(Slfm_track(1:2:end))));
% subplot(222);plot(20*log10(abs(Slfm_track(2:2:end))));
% Slfm_complex=Slfm_track(1:2:end)+j*Slfm_track(2:2:end);%��Ϊ������ʽ
Slfmexp_track=exp(j*2*pi*(F_if*t_track+BandWidth_track/TimeWidth_track/2*t_track.*t_track));%����ʱ���Ե�Ƶ�ź�
Ht_track=conj(fliplr(Slfmexp_track));%ʱ��ƥ���˲�����
Echo=zeros(1,fix(PRT*Fs_ad));DelayNumber=fix(2*R_target/C*Fs_ad);%Ŀ�����12km��Ӧ���ӳ�
Echo(1,(DelayNumber+1):(DelayNumber+length(Slfm_track)))=Slfm_track;%�����ز��ź�
% Echo_noise=1/sqrt(2)*normrnd(0,1,1,size(Echo,2))*10.^(-SNR_IQ/20);%ͨ�������ź�
% Echo=Echo+Echo_noise;%ͨ����������
figure;plot(20*log10(abs(Echo)));grid on;axis([3200,3700,-40,0]);
xlabel('���뵥Ԫ');ylabel('����/dB');title('Ŀ��ز���Ƶ�ź�');
% figure;plot(abs(Echo));grid on;
% xlabel('���뵥Ԫ');ylabel('����');title('Ŀ��ز���Ƶ�ź�');
cos_I=cos(2*pi*F_if/Fs_ad.*[1:length(Echo)]);
sin_Q=sin(2*pi*F_if/Fs_ad.*[1:length(Echo)]);
Echo_I=Echo.*cos_I;
Echo_Q=-Echo.*sin_Q;
I_track=filter(h_lowpass(2:2:end),1,Echo_I(1:end));
Q_track=filter(h_lowpass(1:2:end),1,Echo_Q(1:end));%����IQ�����ź�
figure;subplot(211);plot(20*log10(abs(I_track)));axis([3200,3700,-40,0]);
xlabel('���뵥Ԫ');ylabel('����/dB');title('��������Iͨ�������ź�');grid on;
subplot(212);plot(20*log10(abs(Q_track)));axis([3200,3700,-40,0]);
xlabel('���뵥Ԫ');ylabel('����/dB');title('��������Qͨ�������ź�');grid on;

% figure;subplot(211);plot((abs(I_track)));axis([3200,3700,0,0.5]);
% xlabel('���뵥Ԫ');ylabel('����');title('��������Iͨ�������ź�');grid on;
% subplot(212);plot((abs(Q_track)));axis([3200,3700,0,0.5]);
% xlabel('���뵥Ԫ');ylabel('����');title('��������Qͨ�������ź�');grid on;
Echo_IQ_track=I_track(n_lowpass/2:end)+1j.*Q_track(n_lowpass/2:end);
figure;plot(Echo_IQ_track,'.');axis square;grid on;
xlabel('Iͨ��');ylabel('Qͨ��');title('IQ������');
Echo_IQ_track_pc=conv(Echo_IQ_track,Ht_track);
Echo_IQ_track_pc=Echo_IQ_track_pc(length(Ht_track):end);%�����������Ļ����źŽ�����ѹ����
figure;plot((0:1/(Fs_ad):PRT-((1+fix(n_lowpass/2))/Fs_ad))*C/2*1e-3,20*log10(abs(Echo_IQ_track_pc)));
xlabel('����/km');ylabel('����/dB');title('�����ź���ѹ����');axis([0,80,-20,50]);grid on;axis([8,12,-10,50]);
%  figure;plot((0:1/(Fs_ad):PRT-((1+fix(n_lowpass/2))/Fs_ad))*C/2*1e-3,abs(Echo_IQ_track_pc));
%  xlabel('����/km');ylabel('����');title('�����ź���ѹ����');grid on;

% figure;plot(20*log10(abs(Echo_IQ_track_pc)));
% xlabel('���뵥Ԫ');ylabel('����/dB');title('�����ź���ѹ����');grid on;
%��ѹ��Ŀ���ӳٱ�Ϊ1920��������1929��������1929��1930�϶��м�壬���˲�����Ƶ�ԭ��


