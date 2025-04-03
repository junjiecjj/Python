% https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247485771&idx=1&sn=8e269280b663226160227aec22806c3e&chksm=c11f04def6688dc8e20c2e92bed6bc4547bf107f87b77bfff29f2c66434fdb5702333184ee1d&scene=178&cur_album_id=2442863581802381317#rd
% 雷达信号处理之FMCW 3D-FFT原理（附带MATLAB仿真程序）

clear all;close all;clc;
%% 雷达参数（使用mmWave Studio默认参数）
c=3.0e8;  
B=768e6;       % 调频带宽
K=30e12;       % 调频斜率
T=B/K;         % 采样时间
Tc=160e-6;     % chirp总周期
fs=10e6;       % 采样率
f0=77e9;       % 初始频率
lambda=c/f0;   % 雷达信号波长
d=lambda/2;    % 天线阵列间距
n_samples=256; % 采样点数/脉冲
N=256;         % 距离向FFT点数
n_chirps=128;  % 每帧脉冲数
M=128;         % 多普勒向FFT点数
n_RX=4;        % RX天线通道数
Q = 180;       % 角度FFT
xx = 13;        % 第xx帧
%% 读取回波数据
fname='/home/jack/公共的/RadarData/adc_data.bin';
fid = fopen(fname,'rb');    
%16bits，复数形式(I/Q两路)，4RX,1TX,有符号16bit，小端模式
sdata = fread(fid,xx*n_samples*n_chirps*4*1*2,'int16');    
sdata = sdata((xx-1)*n_samples*n_chirps*4*1*2+1:xx*n_samples*n_chirps*4*1*2);
%% 1642+DCA1000
fileSize = size(sdata, 1);
lvds_data = zeros(1, fileSize/2);
count = 1;
for i=1:4:fileSize-5
   lvds_data(1,count) = sdata(i) + 1i*sdata(i+2); 
   lvds_data(1,count+1) = sdata(i+1)+1i*sdata(i+3); 
   count = count + 2;
end
lvds_data = reshape(lvds_data, n_samples*n_RX, n_chirps);
lvds_data = lvds_data.';
cdata = zeros(n_RX,n_chirps*n_samples);
for row = 1:n_RX
  for i = 1: n_chirps
      cdata(row,(i-1)*n_samples+1:i*n_samples) = lvds_data(i,(row-1)*n_samples+1:row*n_samples);
  end
end
fclose(fid);
data_radar_1 = reshape(cdata(1,:),n_samples,n_chirps);   %RX1
data_radar_2 = reshape(cdata(2,:),n_samples,n_chirps);   %RX2
data_radar_3 = reshape(cdata(3,:),n_samples,n_chirps);   %RX3
data_radar_4 = reshape(cdata(4,:),n_samples,n_chirps);   %RX4
data_radar=[];            
data_radar(:,:,1)=data_radar_1;     %三维雷达回波数据
data_radar(:,:,2)=data_radar_2;
data_radar(:,:,3)=data_radar_3;
data_radar(:,:,4)=data_radar_4;
%% 3维FFT处理
%距离FFT
range_win = hamming(n_samples);   %加海明窗
doppler_win = hamming(n_chirps);
range_profile = [];
for k=1:n_RX
   for m=1:n_chirps
      temp=data_radar(:,m,k).*range_win;    %加窗函数
      temp_fft=fft(temp,N);    %对每个chirp做N点FFT
      range_profile(:,m,k)=temp_fft;
   end
end
%多普勒FFT
speed_profile = [];
for k=1:n_RX
    for n=1:N
      temp=range_profile(n,:,k).*(doppler_win)';    
      temp_fft=fftshift(fft(temp,M));    %对rangeFFT结果进行M点FFT
      speed_profile(n,:,k)=temp_fft;  
    end
end
%角度FFT
angle_profile = [];
for n=1:N   %range
    for m=1:M   %chirp
      temp=speed_profile(n,m,:);    
      temp_fft=fftshift(fft(temp,Q));    %对2D FFT结果进行Q点FFT
      angle_profile(n,m,:)=temp_fft;  
    end
end
%% 绘制2维FFT处理三维视图
figure;
speed_profile_temp = reshape(speed_profile(:,:,1),N,M);   
speed_profile_Temp = speed_profile_temp';
[X,Y]=meshgrid((0:N-1)*fs*c/N/2/K,(-M/2:M/2-1)*lambda/Tc/M/2);
mesh(X,Y,(abs(speed_profile_Temp))); 
xlabel('距离(m)');ylabel('速度(m/s)');zlabel('信号幅值');
title('2维FFT处理三维视图');
xlim([0 (N-1)*fs*c/N/2/K]); ylim([(-M/2)*lambda/Tc/M/2 (M/2-1)*lambda/Tc/M/2]);
%% 计算峰值位置
angle_profile=abs(angle_profile);
pink=max(angle_profile(:));
[row,col,pag]=ind2sub(size(angle_profile),find(angle_profile==pink));

%% 计算目标距离、速度、角度
fb = ((row-1)*fs)/N;         %差拍频率
fd = ((col-M/2-1)*fs)/(N*M); %多普勒频率
fw = (pag-Q/2-1)/Q;          %空间频率
R = c*(fb-fd)/2/K;           %距离公式
v = lambda*fd/2;             %速度公式
theta = asin(fw*lambda/d);   %角度公式
angle = theta*180/pi;

fprintf('目标距离：%f m\n',R);
fprintf('目标速度：%f m/s\n',v);
fprintf('目标角度：%f°\n',angle);