% fig3_MIMO_Radar_A_beampatterns.m
% 产生论文 Fig.3 (a)-(d) 的发射波束方向图
% MIMO Radar A: 发射间距 5λ，接收间距 0.5λ，N=M=10, 目标 -16.5°, 干扰 -5°

clc;
clear;
close all;

%% 系统参数
N = 10;                 % 发射天线数
M = 10;                 % 接收天线数
L = 256;                % 快拍数（波形长度）
P = 1;                  % 总发射功率
ASNR_dB = 40;           % 阵列信噪比 (dB)
sigma2 = (M*N*P) / (10^(ASNR_dB/10));   % 噪声方差

% 干扰机参数
jammer_angle = -5;      % 度
AINR_dB = 100;          % 阵列干扰噪比 (dB)
jammer_power = 10^(AINR_dB/10) * sigma2 / M;

% 目标参数
theta_t = -16.5;        % 目标角度 (deg)
b_t = 1;                % 目标复振幅（实数）

% 阵列几何
d_tx = 5;               % 发射阵元间距（波长）
d_rx = 0.5;             % 接收阵元间距（波长）

% 导向矢量函数
tx_steer = @(theta) exp(1j*2*pi*d_tx*(0:N-1)'*sind(theta));
rx_steer = @(theta) exp(1j*2*pi*d_rx*(0:M-1)'*sind(theta));

% 干扰加噪声协方差矩阵 Q
Q = sigma2 * eye(M) + jammer_power * (rx_steer(jammer_angle) * rx_steer(jammer_angle)');

% 目标方向的导向矢量及导数
v0 = tx_steer(theta_t);
v_dot = 1j*2*pi*d_tx*cosd(theta_t) * (0:N-1)' .* v0;
a0 = rx_steer(theta_t);
a_dot = 1j*2*pi*d_rx*cosd(theta_t) * (0:M-1)' .* a0;

% 预计算常数
A1 = a_dot' * (Q \ a_dot);
A2 = a_dot' * (Q \ a0);
A3 = a0'   * (Q \ a0);   % 正实数

% 计算 alpha, beta (公式 36-37)
alpha = L * abs(b_t)^2 * norm(v0)^2 * (A1 - abs(A2)^2 / A3);
beta  = L * abs(b_t)^2 * norm(v_dot)^2 * A3;

fprintf('alpha = %g, beta = %g\n', alpha, beta);

%% 1. Angle-Only 准则 (α < β 时取小 ζ)
zeta = 1e-6;   % 非常小的正数，使 λ11>0
lambda11_ao = zeta * P;
lambda22_ao = (1 - zeta) * P;
R_ao = (lambda11_ao / norm(v0)^2) * (v0 * v0') + ...
       (lambda22_ao / norm(v_dot)^2) * (v_dot * v_dot');

%% 2. Det-Opt 准则 (闭式解 42)
if beta <= 3*alpha
    coeff1 = 1;
    coeff2 = 0;
else
    coeff1 = (2*beta) / (3*(beta - alpha));
    coeff2 = (beta - 3*alpha) / (3*(beta - alpha));
end
lambda11_det = coeff1 * P;
lambda22_det = coeff2 * P;
R_det = (lambda11_det / norm(v0)^2) * (v0 * v0') + ...
        (lambda22_det / norm(v_dot)^2) * (v_dot * v_dot');

%% 3. Trace-Opt 准则 (CVX SDP)
% 构建 FIM 的线性函数，用于 SDP
% 注意：我们优化 R_Phi，目标为 min trace(C) 即 min trace(F^{-1})
% 等价的 SDP: min trace(Z) s.t. [F, I; I, Z] >= 0, tr(R_Phi)=P, R_Phi>=0
% 这里 F 是 3x3 实 FIM（参数顺序 theta, Re(b), Im(b)）
% 由于 CVX 不支持直接以 F 为变量，需要构造一个函数返回给定 R 的 FIM
% 我们直接编写 SDP

% 预先计算 F 各块关于 R 的线性系数（用 lambda 形式，但需要完整 R）
% 为避免复杂，采用基于矩阵的表达式
% 定义函数 compute_FIM(R)，返回 3x3 实对称 FIM
function F = compute_FIM(R, v0, v_dot, a0, a_dot, Q, b, L)
    % 计算辅助标量
    vRv = real(v0' * R * v0);
    vRvd = v0' * R * v_dot;   % 复数
    vdRvd = real(v_dot' * R * v_dot);
    % 复数 F11, F12, F22
    F11c = L * abs(b)^2 * ( (a_dot'*(Q\a_dot))*vRv + (a_dot'*(Q\a0))*vRvd + ...
            (a_dot'*(Q\a0))'*vRvd' + (a0'*(Q\a0))*vdRvd );
    F12c = L * conj(b) * ( (a_dot'*(Q\a0))*vRv + (a0'*(Q\a0))*vRvd );
    F22c = L * (a0'*(Q\a0)) * vRv;   % 实数
    % 实 FIM
    F = [ real(F11c), -imag(F11c), real(F12c);
          imag(F11c),  real(F11c), imag(F12c);
          real(F12c'),-imag(F12c'), real(F22c) ];
end

% 使用 CVX 求解 Trace-Opt
if exist('cvx_begin', 'file')
    cvx_begin sdp quiet
        variable R_trace(N,N) hermitian
        variable Z(3,3) symmetric
        F_trace = compute_FIM(R_trace, v0, v_dot, a0, a_dot, Q, b_t, L);
        minimize( trace(Z) )
        subject to
            [F_trace, eye(3); eye(3), Z] >= 0;
            trace(R_trace) == P;
            R_trace >= 0;
    cvx_end
else
    warning('CVX not found. Using Det-Opt as placeholder for Trace-Opt.');
    R_trace = R_det;
end

%% 4. Eigen-Opt 准则 (CVX SDP)
% 最大化最小特征值：max t s.t. F >= t*I, tr(R)=P, R>=0
if exist('cvx_begin', 'file')
    cvx_begin sdp quiet
        variable R_eigen(N,N) hermitian
        variable t
        F_eigen = compute_FIM(R_eigen, v0, v_dot, a0, a_dot, Q, b_t, L);
        maximize( t )
        subject to
            F_eigen >= t * eye(3);
            trace(R_eigen) == P;
            R_eigen >= 0;
    cvx_end
else
    warning('CVX not found. Using Det-Opt as placeholder for Eigen-Opt.');
    R_eigen = R_det;
end

%% 计算波束方向图
theta_scan = -90:0.5:90;
bp_ao = zeros(size(theta_scan));
bp_det = zeros(size(theta_scan));
bp_trace = zeros(size(theta_scan));
bp_eigen = zeros(size(theta_scan));

for i = 1:length(theta_scan)
    v = tx_steer(theta_scan(i));
    bp_ao(i) = real(v' * R_ao * v);
    bp_det(i) = real(v' * R_det * v);
    bp_trace(i) = real(v' * R_trace * v);
    bp_eigen(i) = real(v' * R_eigen * v);
end

% 转换为 dB，并归一化（最大值为0dB）
bp_ao_dB = 10*log10(bp_ao / max(bp_ao));
bp_det_dB = 10*log10(bp_det / max(bp_det));
bp_trace_dB = 10*log10(bp_trace / max(bp_trace));
bp_eigen_dB = 10*log10(bp_eigen / max(bp_eigen));

%% 绘图 (Fig.3)
figure('Position', [100,100,1200,800]);

subplot(2,2,1);
plot(theta_scan, bp_ao_dB, 'LineWidth', 1.5);
xlabel('Angle (deg)'); ylabel('Beampattern (dB)');
title('(a) Angle-Only');
xlim([-20,0]); ylim([-40,5]); grid on;

subplot(2,2,2);
plot(theta_scan, bp_eigen_dB, 'LineWidth', 1.5);
xlabel('Angle (deg)'); ylabel('Beampattern (dB)');
title('(b) Eigen-Opt');
xlim([-20,0]); ylim([-40,5]); grid on;

subplot(2,2,3);
plot(theta_scan, bp_trace_dB, 'LineWidth', 1.5);
xlabel('Angle (deg)'); ylabel('Beampattern (dB)');
title('(c) Trace-Opt');
xlim([-20,0]); ylim([-40,5]); grid on;

subplot(2,2,4);
plot(theta_scan, bp_det_dB, 'LineWidth', 1.5);
xlabel('Angle (deg)'); ylabel('Beampattern (dB)');
title('(d) Det-Opt');
xlim([-20,0]); ylim([-40,5]); grid on;

sgtitle('Fig. 3 Optimal Transmit Beampatterns for MIMO Radar A(5,0.5)');
saveas(gcf, 'fig3.png');

fprintf('All beampatterns plotted. Check fig3.png\n');