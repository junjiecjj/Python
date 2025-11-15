%% AFDM-ISAC 一体化仿真
%  - 通信：星座图（发端 vs. 简易等化后的接收端）
%  - 感知：只采用时域 FCCR（III-A：fast-time FFT → 乘积 → IFFT → slow-time FFT）
%  - 可视化：2D/3D Range–Doppler，均叠加真实目标标注；

clc; clear; close all;rng(2025);

%% ===================== 系统参数 =====================
N       = 1024;              % AFDM 子载波数
Nsym    = 64;                % 慢时脉冲数
M       = 16;                % QAM 阶数
SNRdB   = 15;                % AWGN

fc      = 24e9;              % 载频 [Hz]
DeltaF  = 22.729e3;          % 子载波间隔 [Hz]
Fs      = N*DeltaF;          % 采样率 [Hz]
Ts      = 1/Fs;              % 采样间隔 [s]

alphamax = 2; kv = 4;        
c1 = (2*(alphamax+kv)+1)/(2*N);
c2 = 1/(128*N);
Ncp = 128;                   % CPP（需 > 最大时延采样）
TAFDM = (N+Ncp)/Fs;          % 单符号时长（含CPP）

c_light = 3e8;

%% ===================== 发射端：QAM & AFDM 调制 =====================
symIdx = randi([0 M-1], N*Nsym, 1);
xTx = qammod(symIdx, M, 'UnitAveragePower', true);
X   = reshape(xTx, [N, Nsym]);          % DAFT 域符号

s_blocks = afdm_mod(X, c1, c2);         % N x Nsym（时域）
s_tx     = cpp_add(s_blocks, Ncp);      % (N+Ncp)*Nsym x 1

%% ===================== 目标真值（请确保 max(li)<Ncp） =====================
R_true = [400, 320, 180];               % m
v_true = [ 25,  10, -10];               % m/s
P = numel(R_true);

tau = 2*R_true / c_light;               % s
li  = round(tau / Ts);                  % 样点
fd  = 2*v_true*fc / c_light;            % Hz
fi  = fd * Ts;                          % 归一化（乘以 n 的相位增量）

h_i = (randn(1,P)+1j*randn(1,P))/sqrt(2*P);

%% ===================== 接收：叠加回波 + 噪声 =====================
r_rx = apply_radar_channel(s_tx, N, Ncp, Nsym, li, fi, h_i);
r_rx = awgn(r_rx, SNRdB, 'measured');

%% ===================== 通信侧接收：DAFT & 简易等化 =====================
Y_blocks = cpp_remove(r_rx, N, Ncp, Nsym);        % N x Nsym（时域）
Y_daft   = daft_colwise(Y_blocks, c1, c2);        % N x Nsym（DAFT）

Hhat = Y_daft ./ (X + 1e-12);                     % 粗糙 LS，默认通信信息完全已知
Xhat = Y_daft ./ (Hhat + 1e-12);
plot_constellation(X(:), Xhat(:), M);

%% ===================== 公共坐标换算（物理轴） =====================
% 距离分辨率：ΔR = c/(2B) = c/(2*N*Δf) = c/(2*Fs)
range_bin_m = c_light/(2*Fs);

% Doppler 轴：K 点慢时 FFT，Δfd = 1/(K*TAFDM)，范围 ±1/(2*TAFDM)
fd_axis = (-floor(Nsym/2):ceil(Nsym/2)-1) * (1/(Nsym*TAFDM)); % Hz
vel_axis = fd_axis * (c_light/(2*fc));                         % m/s

%% ===================== 感知 ：时域 FCCR =====================
RD_fccr = abs( rdm_fccr(Y_blocks, s_blocks) );     % N x Nsym
RD_show = fftshift(RD_fccr(1:Ncp, :), 2);          % 仅对 Doppler 维中心化
ranges_plot = (0:Ncp-1) * range_bin_m;

% 2D RDM
figure('Name','RDM (Time-Domain FCCR)','Color','w');
imagesc(vel_axis, ranges_plot, mag2db_norm(RD_show)); axis xy;
colormap turbo; colorbar; xlabel('Velocity (m/s)'); ylabel('Range (m)');
title('Time-Domain FCCR (Range–Doppler)'); hold on;
hScat = scatter(v_true, R_true, 80, 'r', 'filled', 'MarkerEdgeColor','k', 'LineWidth',1.2);
legend(hScat, {'True targets'}, 'Location','northoutside'); % 仅真值图例
grid on; box on;

% 3D RDM
figure('Name','RDM 3D (FCCR)','Color','w');
[VV,RR] = meshgrid(vel_axis, ranges_plot);
Zfccr = mag2db_norm(RD_show);
surf(VV, RR, Zfccr, 'EdgeColor','none'); view(40,30);
xlabel('Velocity (m/s)'); ylabel('Range (m)'); zlabel('Magnitude (dB)');
title('3D Range–Doppler (FCCR)'); colormap turbo; colorbar; grid on; box on;
% 3D 叠加真值（Z 取最近网格值）
[~,kidxT] = arrayfun(@(v) min(abs(vel_axis-v)), v_true);
ridxT = min(max(round(R_true/range_bin_m)+1,1), Ncp);
zT = arrayfun(@(ii) Zfccr(ridxT(ii),kidxT(ii)), 1:P);
hold on; scatter3(v_true, R_true, zT, 60, 'r', 'filled', 'MarkerEdgeColor','k', 'LineWidth',1.2);

disp('仿真完成。');

%% ======================================================================
%% ============================ 本地函数区 ===============================
%% ======================================================================

function s_blocks = afdm_mod(X, c1, c2)
% 逐列 IDAFT（论文式(1) 的快速实现）
[N, K] = size(X);
s_blocks = zeros(N, K);
for k = 1:K
    s_blocks(:,k) = idaft_col(X(:,k), c1, c2);
end
end

function s = idaft_col(x, c1, c2)
% 单列 IDAFT
[N, K] = size(x); assert(K==1, 'idaft_col 需要 N×1 列向量');
n  = (0:N-1).'; m  = (0:N-1).';
E1 = exp(1j*2*pi*c1*(n.^2));           % 时间域 chirp
E2 = exp(1j*2*pi*c2*(m.^2));           % DAFT 域 chirp
s  = E1 .* (ifft(E2 .* x, [], 1) * sqrt(N));
end

function Y = daft_colwise(S, c1, c2)
% 逐列 DAFT（IDAFT 的逆）
[N, K] = size(S);
n  = (0:N-1).'; m  = (0:N-1).';
E1 = exp(-1j*2*pi*c1*(n.^2));
E2 = exp(-1j*2*pi*c2*(m.^2));
Y  = (fft(E1 .* S, [], 1) / sqrt(N)) .* E2;
end

function s_tx = cpp_add(s_blocks, Ncp)
% CPP 添加：每块头部拼接 Ncp 个尾样点
[N, K] = size(s_blocks);
out = zeros(N+Ncp, K);
for k = 1:K
    blk = s_blocks(:,k);
    out(:,k) = [blk(end-Ncp+1:end); blk];
end
s_tx = out(:);
end

function S_blocks = cpp_remove(rx, N, Ncp, K)
% 去 CPP：按块切分并去掉前 Ncp
rx_mat = reshape(rx, N+Ncp, K);
S_blocks = rx_mat(Ncp+1:end, :);
end

function r = apply_radar_channel(s_tx, N, Ncp, K, li, fi, h_i)
% 论文式(3)：r[n] = Σ_i h_i · s[n-li] · e^{j2π f_i n}
P = numel(li);
L = (N+Ncp)*K;
n = (0:L-1).';
r = zeros(L,1);
s_mat = reshape(s_tx, N+Ncp, K);
for p = 1:P
    s_shift = zeros(size(s_mat));
    for k = 1:K
        s_shift(:,k) = circshift(s_mat(:,k), li(p));
    end
    r = r + h_i(p) * ( s_shift(:) .* exp(1j*2*pi*fi(p)*n) );
end
end

function RD = rdm_fccr(R, S)
% 时域 FCCR：fast-time FFT → 乘积 → IFFT → slow-time FFT
Rf = fft(R, [], 1);
Sf = fft(S, [], 1);
Z  = ifft(Rf .* conj(Sf), [], 1);
RD = fft(Z, [], 2);
end

function ydb = mag2db_norm(Y)
% 归一幅度 → dB（避免 -Inf）
Yn = Y ./ (max(Y(:)) + eps);
ydb = 20*log10(Yn + eps);
end

function plot_constellation(X, Xhat, M)
% 星座图（仅演示）
Nplot = min(4000, numel(X));
idx = randperm(numel(X), Nplot);
figure('Name','Constellation','Color','w');
subplot(1,2,1);
plot(real(X(idx)), imag(X(idx)), 'o'); axis equal; grid on;
title(sprintf('TX %d-QAM', M)); xlabel('I'); ylabel('Q');
subplot(1,2,2);
plot(real(Xhat(idx)), imag(Xhat(idx)), 'o'); axis equal; grid on;
title('RX Equalized'); xlabel('I'); ylabel('Q');
end
