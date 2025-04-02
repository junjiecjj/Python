clear; clc; close all; warning off;
% https://zhuanlan.zhihu.com/p/687473210
% 第4讲 -- 线性调频连续波LFMCW测量原理：测距、测速、测角

%% 参数设置
maxR = 200;  % 最大探测距离
rangeRes = 1;  % 距离分辨率
maxV = 70;  % 最大检测目标的速度
fc = 77e9;  % 工作频率（载频）
c = 3e8;  % 光速
r0 = 90;  % 目标距离设置 (max = 200m)
v0 = 10;  % 目标速度设置 (min =-70m/s, max=70m/s)

%% 产生信号
B = c / (2 * rangeRes);  % 扫频带宽（B = 150MHz）
Tchirp = 5.5 * 2 * maxR / c;  % 扫频时间 (x-axis), 5.5= sweep time should be at least 5 o 6 times the round trip time
S = B / Tchirp;  % 调频斜率
phi = 0;  % 初相位
N_chrip = 128;  % chirp数量           
Ns = 4096;  % ADC采样点数
t = linspace(0, N_chrip * Tchirp, N_chrip * Ns);  % 发射信号和接收信号的采样时间
ft = fc .* t + (S .* t.^2) ./ 2;
Tx = cos(2 * pi .* ft + phi);  % 发射信号
tau = Tchirp / 6;  % 时延
fr = fc .* (t - tau) + S .* (t - tau).^2;  % 回波信号频率
Rx = cos(2 * pi .* fr / 2 + phi);  % 回波信号

%% 经过混频器
Mix = Tx .* Rx;

%% 混频经过低通滤波器
fpass = 30e5;  % 截止频率fpass=30MHz
fs_lpf = 120e6;  % 采样频率fs=120MHz
Mix_filtered = lowpass(Mix(1:Ns), fpass, fs_lpf); 

%% 计算差频
N_fft = 1024;
f = (0 : N_fft - 1) / 2 * Ns;
Mix_filtered_fft = db(abs(fft(Mix_filtered, N_fft)));

%% 作图
figure(1); clf;
sp1 = subplot(2, 2, 1); plot(t(1:Ns), Tx(1:Ns), 'linewidth', 1.2); axis('tight');
xlabel('时间'); ylabel('幅值'); title('发射信号时域波形图'); set(gca, 'fontsize', 12);
sp3 = subplot(2, 2, 3); plot(t(1:Ns), Rx(1:Ns), 'linewidth', 1.2); axis('tight'); 
xlabel('时间'); ylabel('幅值'); title('回波信号时域波形图'); set(gca, 'fontsize', 12);
subplot(2, 2, [2, 4]); plot(t(1:Ns), ft(1:Ns), 'linewidth', 1.2); hold on;
plot(t(1:Ns), fr(1:Ns)); hold off; axis('tight'); ylim([0, max(fr(1:Ns))]);
xlabel('时间'); ylabel('频率'); legend('发射信号频率', '回波信号频率'); set(gca, 'fontsize', 12);
linkaxes([sp1, sp3], 'x'); 
figure(2); 
sp11 = subplot(3, 1, 1); plot(t(1:Ns), Mix(1:Ns), 'linewidth', 1.2); axis('tight');
xlabel('时间'); ylabel('幅值'); title('混频信号'); set(gca, 'fontsize', 12);
sp22 = subplot(3, 1, 2); plot(t(1:Ns), Mix_filtered(1:Ns), 'linewidth', 1.2); axis('tight'); 
xlabel('时间'); ylabel('幅值'); title('差频信号'); set(gca, 'fontsize', 12);
subplot(3, 1, 3); plot(f, Mix_filtered_fft, 'linewidth', 1.2); axis('tight'); 
xlabel('频率'); ylabel('幅度（dB）'); title('差频信号的频谱图'); set(gca, 'fontsize', 12);
linkaxes([sp11, sp22], 'x');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 雷达参数设计
c = 3e8;  % 光速
fc = 77e9;  % 载频（77GHz）
lambda =  c / fc;  % 波长
B = 4e9;  % 扫频带宽（4GHz）
Tchrip = 20e-6;  % 扫频时宽（20us）
Nchrip = 256;  % 一个frame的chrip数量
fs = Nchrip / Tchrip;  % 采样率
Ns = 1024;  % 单个chrip周期内的采样点数
Tframe = Tchrip * Ns;  % 一个frame持续时长
S = B / Tchrip;  % 扫频斜率
Rx_num = 1;  % 接收天线数量

%% 测量参数计算
d_res = c / (2 * B);  % 距离分辨率
d_max = (c * fs) / (2 * S);  % 最大探测距离
v_max = lambda / (4 * Tchrip);  % 最大不模糊速度
v_res = lambda / (2 * Nchrip * Tchrip);  % 速度分辨率

%% 设置目标参数（单个目标）
d0 = 5;  % 目标位置
v0 = 15;  % 目标速度
rcs = 10;  % 目标RCS
sigma = 0.1;  % 高斯白噪声标准差

%% 产生混频信号
t = linspace(0, Tchrip, Ns);
ft = fc .* t + S .* t.^2 / 2;
St = cos(2 * pi .* ft);
Smix_frame = zeros(Nchrip, Ns);
for chrip = 1 : Nchrip
    d = d0 + v0 * (t + (chrip - 1) * Tchrip);
    tau = 2 .* d ./ c;  % 运动目标的时延是动态变化的
    Sr = zeros(1, Ns);
    for target = 1 : length(d0)
        fr = fc .* (t - tau(target, :)) + S * (t - tau(target, :)).^ 2 / 2;
        Sr = Sr + rcs(target) * cos(2 * pi * fr) + wgn(1, Ns, sigma(target));  % 总的回波信号=所有目标的回波信号之和
    end   
    Smix = St .* Sr;
    Smix_frame(chrip, :) = Smix;
end

%% 距离多普勒分析
Nfft_v = Nchrip * 2;
Nfft_d = Ns;
x = (0 : Nfft_d - 1) / Nfft_d * Ns * d_res;  % RDM横轴转换为距离
y = linspace(-v_max, v_max, Nfft_v);  % RDM横轴转换为速度
FFT_2D = abs(fftshift(fft2(Smix_frame, Nfft_v, Nfft_d), 1)); 

figure(3); 
subplot(1, 2, 1); mesh(x(1:Nfft_d/2), y, FFT_2D(:, 1:Nfft_d/2)); axis('tight'); set(gca, 'fontsize', 12);
xlabel('距离（m）'); ylabel('速度（m/s）'); title('距离-多普勒2D-FFT（fft2）');
subplot(1, 2, 2); mesh(x(1:Nfft_d/2), y, FFT_2D(:, 1:Nfft_d/2)); axis('tight'); set(gca, 'fontsize', 12); view(2);
xlabel('距离（m）'); ylabel('速度（m/s）'); title('俯视图（fft2）');






