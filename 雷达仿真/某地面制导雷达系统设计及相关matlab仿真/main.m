clear; close all; clc;

% https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247489532&idx=1&sn=e904c1434d46eb1e070c49cf76dc5636&chksm=ceecd3d6d82564724dbebec9178718ed5de71897011ff1a630e56cad01f3e59f3963110e0189&mpshare=1&scene=1&srcid=1227rBdDjX9ekVTlvRsYKRQk&sharer_shareinfo=673f009a182a77ebaf0508744899101d&sharer_shareinfo_first=673f009a182a77ebaf0508744899101d&exportkey=n_ChQIAhIQU1cd%2FLjsoos0Gphr3y1B1xKfAgIE97dBBAEAAAAAAMELGFeoAfMAAAAOpnltbLcz9gKNyK89dVj0nUPn8E10O0wxImziN7alHMiEQuBykTB7rkVj0NH08Br1uW%2Fh7kbr4SUXmR8Z9nGoAr0GNByWnRIimVm%2B%2FtWZwWrbZV%2FSRLAo6jq9ICAFvI5FEYsdltKj93k8%2FIAmWOpmQBcxe%2BNGpPYir8ZMIGi1IWv8DTeCuf%2F8YV3Suyh2HYIs1h%2F%2FiwLaIEJM2sYXF6YdunrpYOxkUDWiLHisIHpxpQLsQzLZ1Ejedn9hHniVUAUMq2Sf%2B3TpU8ysVKScqMpUo32KMwwXqtpEzgEY%2BvFk7Ebr%2BI3NF4EH3uevV%2BC9TUKzrBDxUf%2Bn5tI%2FEmnwwVlabp4JYek6ASWw&acctmode=0&pass_ticket=4LdMKQCdpZsA6pUSFB8%2FTliDzxJ%2BKZ8ROTjvsUVQl6JfWCcsqkUi1GdMBbGCLR2e&wx_header=0#rd
%% 1. 定义参数（来自之前的雷达系统参数）
clear; clc; close all;
c = 3e8;           % 光速 (m/s)
f = 3e9;           % 雷达工作频率 (Hz)
lambda = c/f;      % 波长 (m)，计算得0.1m
G = 2827;          % 天线增益（线性值，对应34.5dB）
k = 1.38e-23;      % 玻尔兹曼常数 (J/K)
T0 = 290;          % 室温 (K)
F = 10^(6/10);     % 噪声系数（线性值，6dB对应3.9811）
L = 10^(8/10);     % 损失因子（线性值，8dB对应6.3096）
Pt = 1858.6e3;     % 单个脉冲峰值功率 (W，1858.6kW)
M = 7;             % 积累脉冲数
B=2e6;             %带宽
I_NC = 10^(7.25/10); % 非相干积累改善因子（7.25dB对应线性值≈5.318）
sigma_m = 0.1;     % 导弹RCS (-10dBsm对应0.1m²)
sigma_a = 4;       % 飞机RCS (6dBsm对应4m²)
R = linspace(3e4, 1e5, 1000); % 距离范围：30km ~ 100km（1000个采样点）
%% 2. 计算不同情况下的SNR（线性值）
% 单个脉冲（不积累）的SNR
SNR_single_m = (Pt * G^2 * lambda^2 * sigma_m) ./ ((4*pi)^3 * k * T0 * F * L *B * R.^4);
SNR_single_a = (Pt * G^2 * lambda^2 * sigma_a) ./ ((4*pi)^3 * k * T0 * F * L *B * R.^4);
% 非相干积累后的SNR
SNR_NC_m = SNR_single_m * I_NC;
SNR_NC_a = SNR_single_a * I_NC;
% 相干积累后的SNR（相干积累改善因子为M）
SNR_CI_m = SNR_single_m * M;
SNR_CI_a = SNR_single_a * M;
%% 3. 转换为dB（更符合工程习惯）
SNR_single_m_dB = 10*log10(SNR_single_m);
SNR_single_a_dB = 10*log10(SNR_single_a);
SNR_NC_m_dB = 10*log10(SNR_NC_m);
SNR_NC_a_dB = 10*log10(SNR_NC_a);
SNR_CI_m_dB = 10*log10(SNR_CI_m);
SNR_CI_a_dB = 10*log10(SNR_CI_a);
%% 4. 绘图
figure(1);
figure('Color','w');
hold on; grid on;
plot(R/1000, SNR_single_m_dB, 'r--', 'LineWidth',1.5); % 导弹-不积累
plot(R/1000, SNR_NC_m_dB, 'r-', 'LineWidth',1.5);   % 导弹-非相干积累
plot(R/1000, SNR_CI_m_dB, 'r:', 'LineWidth',1.5);   % 导弹-相干积累
plot(R/1000, SNR_single_a_dB, 'b--', 'LineWidth',1.5);% 飞机-不积累
plot(R/1000, SNR_NC_a_dB, 'b-', 'LineWidth',1.5);    % 飞机-非相干积累
plot(R/1000, SNR_CI_a_dB, 'b:', 'LineWidth',1.5);    % 飞机-相干积累
% 标注检测门限（15dB）
plot(R/1000, 15*ones(size(R)), 'k-', 'LineWidth',1, 'DisplayName','检测门限(15dB)');
xlabel('距离 R (km)','FontSize',10);
ylabel('信噪比 SNR (dB)','FontSize',10);
title('飞机/导弹在积累/不积累下的SNR-距离关系','FontSize',12);
legend('导弹-不积累','导弹-非相干积累','导弹-相干积累',...
       '飞机-不积累','飞机-非相干积累','飞机-相干积累','检测门限(15dB)','Location','southeast');



%%%%% 步骤2：计算目标起伏带来的影响
clear; clc; close all;
% 1. 定义核心检测参数
Pfa = 1e-7;          % 虚警概率
Pd_target = 0.99;    % 目标检测概率（要求≥0.99）
N = 7;               % 积累脉冲数（与之前一致，M=7）
SNR0_dB = 15.00;     % 强制固定非起伏目标基准SNR为15.00 dB
S0 = 10^(SNR0_dB / 10); % 转换为线性值
% 2. 第一步：求解检测门限V_T（由虚警概率Pfa确定，二分法）
% 虚警概率计算公式（N脉冲非相干积累，正态噪声背景）
% Pfa = exp(-Vt) * Σ(Vt^k / k!) ，k从0到N-1
pfa_calc = @(Vt) exp(-Vt) * sum(arrayfun(@(k) Vt^k / factorial(k), 0:N-1));
% 二分法求解满足Pfa的检测门限V_T
Vt_lower = 5;    % 下界初始化
Vt_upper = 25;   % 上界初始化
tol = 1e-6;      % 计算精度
while Vt_upper - Vt_lower > tol
    Vt_mid = (Vt_lower + Vt_upper) / 2;
    Pfa_mid = pfa_calc(Vt_mid);
    if Pfa_mid > Pfa
        % 虚警概率偏大，需要提高门限
        Vt_lower = Vt_mid;
    else
        % 虚警概率偏小，需要降低门限
        Vt_upper = Vt_mid;
    end
end
V_T = Vt_mid;
fprintf('检测门限 V_T：%.4f\n', V_T);
fprintf('非起伏目标所需SNR（基准，强制固定）：%.2f dB\n', SNR0_dB);
% 3. 第二步：计算SwerlingⅠ型（飞机，慢起伏）所需SNR及额外SNR
% SwerlingⅠ型检测概率公式（N脉冲非相干积累，慢起伏目标）
pd_swerling1 = @(S) 1 - exp(-S) * sum(arrayfun(@(k) (S^k / factorial(k)) .* (1 + S/N).^(-(k+1)), 0:N-1));
% 二分法求解SwerlingⅠ型所需SNR
S1_lower = 50;
S1_upper = 90;
while S1_upper - S1_lower > tol
    S1_mid = (S1_lower + S1_upper) / 2;
    Pd_mid = pd_swerling1(S1_mid);
    if Pd_mid < Pd_target
        S1_lower = S1_mid;
    else
        S1_upper = S1_mid;
    end
end
S1 = S1_mid;
SNR1_dB = 10 * log10(S1);
extra_SNR_S1 = SNR1_dB - SNR0_dB; % 飞机所需额外SNR（起伏损失）
fprintf('SwerlingⅠ型（飞机）所需SNR：%.2f dB，额外SNR：%.2f dB\n', SNR1_dB, extra_SNR_S1);
% 4. 第三步：计算SwerlingⅢ型（导弹，快起伏）所需SNR及额外SNR
% SwerlingⅢ型检测概率公式（N脉冲非相干积累，快起伏目标）
pd_swerling3 = @(S) 1 - (1 + S/(2*N)).^(-N);
% 二分法求解SwerlingⅢ型所需SNR
S3_lower = 45;
S3_upper = 85;
while S3_upper - S3_lower > tol
    S3_mid = (S3_lower + S3_upper) / 2;
    Pd_mid = pd_swerling3(S3_mid);
    if Pd_mid < Pd_target
        S3_lower = S3_mid;
    else
        S3_upper = S3_mid;
    end
end
S3 = S3_mid;
SNR3_dB = 10 * log10(S3);
extra_SNR_S3 = SNR3_dB - SNR0_dB; % 导弹所需额外SNR（起伏损失）
fprintf('SwerlingⅢ型（导弹）所需SNR：%.2f dB，额外SNR：%.2f dB\n', SNR3_dB, extra_SNR_S3);
% 6. 可视化结果（直观对比）
figure(2);
figure('Color', 'w', 'Position', [100, 100, 600, 400]);
% 定义目标类型和对应SNR
target_types = categorical({'SwerlingⅠ（飞机）','SwerlingⅢ（导弹）','非起伏（基准）'});
required_SNR = [SNR1_dB; SNR3_dB;SNR0_dB];
extra_SNR = [ extra_SNR_S1; extra_SNR_S3;0]; % 基准额外SNR为0
% 绘制所需SNR柱状图
bar(target_types, required_SNR, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'black');
hold on;
% 绘制非起伏基准线（15dB）
plot([0.5, 3.5], [SNR0_dB, SNR0_dB], 'r--', 'LineWidth', 1.5, 'DisplayName', '非起伏基准线（15dB）');
% 标注额外SNR
for i = 1:length(target_types)
    if i > 0 
        text(i, required_SNR(i) + 0.8, sprintf('额外%.2f dB', extra_SNR(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'red');
    end
    % 标注所需SNR数值
    text(i, required_SNR(i) - 1, sprintf('%.2f dB', required_SNR(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'white');
end
xlabel('目标RCS起伏类型', 'FontSize', 11);
ylabel('所需SNR（dB）', 'FontSize', 11);
title('Swerling目标起伏对应的SNR需求及额外SNR（Pd=0.99, Pfa=1e-7, N=7）', 'FontSize', 12);
grid on;
legend('Location', 'southeast');
hold off;


%%%% 步骤3：计算模糊图
%线性调频脉冲信号模糊函数
clear; clc; close all;
B = 2e6;                %信号带宽
Tp = 80e-6;              %脉冲宽度
Grid = 512;             %坐标轴点数
u = B/Tp;               %调频斜率
t=-Tp:Tp/Grid:Tp-Tp/Grid;
f=-B:B/Grid:B-B/Grid;  
[tau,fd]=meshgrid(t,f);
var1 = Tp - abs(tau);
var2 = pi*(fd - u*tau).*var1;
amf = abs(sinc(var2).*var1/Tp);
amf = amf/max(max(amf));
var3 = pi*u*tau.*var1;
tau1 = abs(sinc(var3).*var1);
tau1 = tau1/max(max(tau1));            %归一化距离模糊
mul = Tp.*abs(sinc(pi*fd.*Tp));
mul = mul/max(max(mul));               %归一化速度模糊
figure(3);
surfl(tau/max(max(tau)),fd/max(max(fd)),amf);
grid on;
xlabel('时间/us');
ylabel('fd/MHz');
zlabel('幅值');
title('线性调频脉冲信号的模糊函数');
figure(4)
contour(tau*1e6,fd*1e-6,amf,1,'b ');grid on;
grid on;axis([-2 2 -4 4]);
xlabel('时间/us');
ylabel('fd/MHz');
title('线性调频脉冲信号的模糊度图');
figure(5);
plot(t*1e6,tau1(Grid+1,:));
grid on;axis([-2 2 0 1]);
xlabel('时间/us');ylabel('|x(t,0)|');
title('距离模糊函数 ')
figure(6);
plot(fd*1e-6,mul(:,Grid+1));
grid on;axis([-4 4 0 1]);
xlabel('fd/MHz');ylabel('|x(0,fd)|');
title('速度模糊函数 ')

%%%% 步骤4：设计脉冲压缩
clear; clc; close all;
%**************参数配置*********************
if 0
Tp=200e-6;          %发射脉冲宽度s
B=1e6;           %调频带宽Hz
Ts=0.5e-6;       %采样时钟s
R0=[80e3,85e3];      %目标的距离m
vr=[0,0];            %目标速度
SNR=[20 20];         %信噪比
Rmin=20e3;           %采样的最小距离
Rrec=150e3;          %接收距离窗的大小
bos=2*pi/0.03;       %波数2*pi/λ。
else
c = 3e8;               % 光速 (m/s)
fc = 3e9;              % 载波频率 (3GHz)
lambda = 0.1;          %波长0.1m
Tp=80e-6;          %发射脉冲宽度s
B=2e6;           %调频带宽Hz
Ts=0.25e-6;       %采样时钟s
R0=[75e3,75.15e3];      %目标的距离m
vr=[0,0];            %目标速度
SNR=[10 10];         %信噪比
Rmin=30e3;           %采样的最小距离
Rrec=105e3;          %接收距离窗的大小
bos=2*pi/lambda;       %波数2*pi/λ。
end
%*********************************************
mu=B/Tp;            %调频率
c=3e8;              %光速m/s
M=round(Tp/Ts);
t1=(-M/2+0.5:M/2-0.5)*Ts;       %时间矢量
NR0=ceil(log2(2*Rrec/c/Ts));NR1=2^NR0;    %计算FFT的点数
lfm=exp(j*pi*mu*t1.^2);                   %信号复包络
lfm_w=lfm.*hanning(M)';
gama=(1+2*vr./c).^2;                      
sp=0.707*(randn(1,NR1)+j*randn(1,NR1));        %噪声
for k= 1:length(R0)
    NR=fix(2*(R0(k)-Rmin)/c/Ts);
    spt=(10^(SNR(k)/20)) *exp(-j*bos*R0(k))*exp(j*pi*mu*gama(k)*t1.^2);     %信号
    sp(NR:NR+M-1)=sp(NR:NR+M-1)+spt;
end
spf=fft(sp,NR1);
lfmf=fft(lfm,NR1);      %未加窗
lfmf_w=fft(lfm_w,NR1);      %加窗
y=abs(ifft(spf.*conj(lfmf),NR1)/NR0);
y1=abs(ifft(spf.*conj(lfmf_w),NR1)/NR0);   %加窗
figure(7);
plot(real(sp));grid on;
xlabel('时域采样点');
figure(8);
plot(t1*1e6,real(lfm));grid on;
xlabel('时间/us')
ylabel('匹配滤波系数实部')
figure(9);
plot((0:NR1-1)*0.0625,20*log10(y));grid on;
xlabel("距离/km")
ylabel("脉压输出/dB");
title("脉冲压缩结果（未加窗）")
figure(10);
plot((0:NR1-1)*0.0625,20*log10(y1));grid on;
xlabel("距离/km")
ylabel("脉压输出/dB");
title("脉冲压缩结果（加窗）")

%%%% 步骤5：计算改善因子

% 1. 清空环境+定义核心参数（来自前文）
clear; clc; close all;
% 雷达/杂波参数
fr = 1000;               % 脉冲重复频率 (Hz)
Tr = 1/fr;               % 脉冲重复周期 (s)
sigma_w = 6.4;           % 风速引起的杂波谱宽 (Hz)
sigma_s = 35.9;          % 天线扫描引起的杂波谱宽 (Hz)
sigma_Bc = sqrt(sigma_w^2 + sigma_s^2); % 杂波总均方根谱宽 (Hz) ≈36.5Hz
N = 10000;               % 杂波样本数（保证统计特性稳定）
% 2. 生成符合谱宽的杂波信号（高斯谱杂波）
Fs = fr;                 % 采样频率=PRF（MTI按脉冲周期采样）
f = linspace(-Fs/2, Fs/2, N); % 频率轴
% 生成高斯型杂波谱（匹配sigma_Bc的谱宽）
clutter_spectrum = exp(-f.^2/(2*sigma_Bc^2)); 
clutter_spectrum = clutter_spectrum / max(clutter_spectrum); % 归一化
% 随机相位法生成时域杂波
phase = 2*pi*rand(1, N); % 随机相位
complex_spec = clutter_spectrum .* exp(1j*phase);
clutter_time = ifftshift(ifft(ifftshift(complex_spec))); % IFFT得到时域信号
clutter_time = real(clutter_time); % 取实部（杂波为实信号）
% 3. 实现2/3/4脉冲MTI杂波抑制（延迟线对消器）
% 2脉冲MTI（一阶对消：x(n) - x(n-1)）
mti2_out = clutter_time(2:end) - clutter_time(1:end-1);
% 3脉冲MTI（二阶对消：x(n) - 2x(n-1) + x(n-2)）
mti3_out = clutter_time(3:end) - 2*clutter_time(2:end-1) + clutter_time(1:end-2);
% 4脉冲MTI（三阶对消：x(n) - 3x(n-1) + 3x(n-2) - x(n-3)）
mti4_out = clutter_time(4:end) - 3*clutter_time(3:end-1) + 3*clutter_time(2:end-2) - clutter_time(1:end-3);
% 4. 计算改善因子（理论值+仿真值）
% 输入杂波功率（方差表示功率）
power_in = var(clutter_time);
% 输出杂波功率
power_out2 = var(mti2_out);
power_out3 = var(mti3_out);
power_out4 = var(mti4_out);
% 仿真改善因子（输入功率/输出功率）
I2_sim = power_in / power_out2;
I3_sim = power_in / power_out3;
I4_sim = power_in / power_out4;
% 理论改善因子
I2_theo = 2*(fr/(2*pi*sigma_Bc))^2;
I3_theo = 2*(fr/(2*pi*sigma_Bc))^4;
I4_theo = (4/3)*(fr/(2*pi*sigma_Bc))^6;
% 5. 输出结果（数值+dB转换）
fprintf('=== MTI杂波抑制改善因子 ===\n');
fprintf('2脉冲MTI：理论=%.0f (%.1f dB)，仿真=%.0f (%.1f dB)\n', ...
    I2_theo, 10*log10(I2_theo), I2_sim, 10*log10(I2_sim));
fprintf('3脉冲MTI：理论=%.0f (%.1f dB)，仿真=%.0f (%.1f dB)\n', ...
    I3_theo, 10*log10(I3_theo), I3_sim, 10*log10(I3_sim));
fprintf('4脉冲MTI：理论=%.0f (%.1f dB)，仿真=%.0f (%.1f dB)\n', ...
    I4_theo, 10*log10(I4_theo), I4_sim, 10*log10(I4_sim));
% 6. 可视化结果（杂波时域+MTI处理后波形）
figure(11);
% 子图1：原始杂波时域波形
subplot(2,2,1);
plot(1:200, clutter_time(1:200)); % 取前200点展示
xlabel('脉冲数'); ylabel('杂波幅值');
title('原始杂波时域波形（前200点）');
grid on;
% 子图2：2脉冲MTI处理后波形
subplot(2,2,2);
plot(1:199, mti2_out(1:199));
xlabel('脉冲数'); ylabel('MTI输出幅值');
title('2脉冲MTI处理后波形（前199点）');
grid on;
% 子图3：3脉冲MTI处理后波形
subplot(2,2,3);
plot(1:198, mti3_out(1:198));
xlabel('脉冲数'); ylabel('MTI输出幅值');
title('3脉冲MTI处理后波形（前198点）');
grid on;
% 子图4：4脉冲MTI处理后波形
subplot(2,2,4);
plot(1:197, mti4_out(1:197));
xlabel('脉冲数'); ylabel('MTI输出幅值');
title('4脉冲MTI处理后波形（前197点）');
grid on;
% 7. 杂波谱+MTI处理后谱对比（验证抑制效果）
figure(12);
% 原始杂波谱
spec_in = 20*log10(abs(fftshift(fft(clutter_time)))/max(abs(fft(clutter_time))));
% 2脉冲MTI处理后谱
spec2 = 20*log10(abs(fftshift(fft(mti2_out)))/max(abs(fft(mti2_out))));
% 3脉冲MTI处理后谱
spec3 = 20*log10(abs(fftshift(fft(mti3_out)))/max(abs(fft(mti3_out))));
% 4脉冲MTI处理后谱
spec4 = 20*log10(abs(fftshift(fft(mti4_out)))/max(abs(fft(mti4_out))));
f_axis = linspace(-fr/2, fr/2, length(spec_in));
plot(f_axis, spec_in, 'k-', 'LineWidth',1.2); hold on;
plot(f_axis(2:end), spec2, 'r--', 'LineWidth',1.2);
plot(f_axis(3:end), spec3, 'b-.', 'LineWidth',1.2);
plot(f_axis(4:end), spec4, 'g:', 'LineWidth',1.2);
xlabel('频率 (Hz)'); ylabel('幅值 (dB)');
title('杂波谱+MTI处理后谱对比');
legend('原始杂波谱','2脉冲MTI谱','3脉冲MTI谱','4脉冲MTI谱');
grid on; ylim([-60, 0]);

















