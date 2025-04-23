

clc;clear; close all;
%%%%%%%% MUSIC CRLB for Uniform Linear Array%%%%%%%%
derad = pi/180;                                                             %角度->弧度
M = 4;                                                                      % 阵元个数
K = 1;                                                                      % 信源数目
N = 1;                                                                      % 快拍数
delay=0.1*pi;                                                               %待估计的延时，可以由测向信号产生
snr=10:2:30;                                                                %遍历信噪比范围
trail=10000;                                                                %尝试次数
d=0:1:(M-1);                                                                %线阵间隔
esti_music=zeros(1,trail);
Len_SNR=length(snr);
VAREsti_snr=zeros(Len_SNR,1);
VARCalc_snr=zeros(Len_SNR,1);
for index_snr=1:Len_SNR
    snr_current=snr(index_snr)
    A=exp(-j*d.'*delay);                                                    %方向矢量，复数形式

    for index_trail=1:trail
        S=randn(K, N);                                                       %信源信号，入射信号，不相干即可，也可以用正弦替代
        X=A*S;                                                              %构造接收信号
        X1=awgn(X, snr_current, 'measured');                                  %引入高斯白噪声，此时的SNR为对数形式
        Rxx=X1*X1'/N;                                                       %标准MUSIC算法，计算协方差矩阵
        [EV,D] = eig(Rxx);                                                    %特征值分解
        EVA = diag(D)';                                                       %将特征值矩阵对角线提取并转为一行
        [EVA,I] = sort(EVA);                                                  %将特征值排序 从小到大
        EV = fliplr(EV(:,I));                                                 % 对应特征矢量排序
        iang = -pi:0.01*pi:pi;                                                 %延时遍历范围
        for index = 1:length(iang)
            angle_input = iang(index);
            phim=angle_input;
            a=exp(-j*d*phim).';
            En=EV(:,K+1:M);                                                 % 取矩阵的第M+1到N列组成噪声子空间
            Pmusic(index)=1/(a'*En*En'*a);                                  %行程MUSIC谱
        end
        Pmusic=abs(Pmusic);
        [y x]=max(Pmusic);                                                  %单目标，求最大值即可
        esti_music(index_trail)=iang(x(1));                                 %寻找最大值对应延时
    end

    VAREsti_snr(index_snr)=(sum((esti_music-delay).^2)/(trail));                %计算方差
    SNR_Linear=10^(snr_current/10);                                         %将对数SNR转换成线性格式
    VARCalc_snr(index_snr)=6/(N*M*(M-1)*(M+1)*SNR_Linear);

end

figure(1)                                                                   %线性CRLB作图，近似为1/x函数
plot(snr,VAREsti_snr,'b')
hold on
plot(snr,VARCalc_snr,'r')
hold off

figure(2)                                                                   %对数CRLB作图，近似斜率为负的线性函数
plot(snr,10*log10(VAREsti_snr),'b')
hold on
plot(snr,10*log10(VARCalc_snr),'r')
hold off
