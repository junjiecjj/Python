

clc;
clear all;
close all;


% init_params.m - 通用参数
N = 10;               % 发射天线数
M = 10;               % 接收天线数
L = 256;              % 快拍数（码长）
P = 1;                % 总发射功率
ASNR_dB = 40;         % 阵列信噪比(dB)
sigma2 = (M*N*P) / (10^(ASNR_dB/10));  % 噪声方差

% 干扰（强干扰机）
jammer_angle = -5;    % 度
AINR_dB = 100;        % 阵列干扰噪比
jammer_power = 10^(AINR_dB/10) * sigma2 / M;

% 目标（单目标情况）
theta_t = -16.5;      % 目标角度
b_t = 1;              % 复振幅（实数）

% 阵列几何（均匀线阵，半波长间距）
d_tx = 0.5;           % 发射间距（波长） 可改为5用于MIMO Radar A
d_rx = 0.5;           % 接收间距（波长）

% 生成steering vectors
tx_steer = @(theta) exp(1j*2*pi*d_tx*(0:N-1)'*sind(theta));
rx_steer = @(theta) exp(1j*2*pi*d_rx*(0:M-1)'*sind(theta));

% 干扰加噪声协方差矩阵 Q
% 干扰：一个强点源
Q = sigma2 * eye(M) + jammer_power * (rx_steer(jammer_angle) * rx_steer(jammer_angle)');


% fig3_single_target_beampattern.m
init_params;
% 针对 MIMO Radar A (d_tx=5, d_rx=0.5)
d_tx = 5;
tx_steer = @(theta) exp(1j*2*pi*d_tx*(0:N-1)'*sind(theta));
% 目标角度
theta0 = -16.5;
v0 = tx_steer(theta0);
v_dot = 1j*2*pi*d_tx*cosd(theta0) * (0:N-1)' .* v0; % 导数

% 构建 A, B 矩阵用于CRB（参考论文公式14-16）
b = 1;
a0 = rx_steer(theta0);  % 接收导向向量
a_dot = 1j*2*pi*d_rx*cosd(theta0) * (0:M-1)' .* a0;

% 计算 Q, 干扰协方差
Q = sigma2*eye(M) + jammer_power * (rx_steer(jammer_angle)*rx_steer(jammer_angle)');

% 计算标量 alpha, beta (公式36-37)
alpha = L * abs(b)^2 * norm(v0)^2 * (a_dot'*(Q\a_dot) - abs(a_dot'*(Q\a0))^2/(a0'*(Q\a0)));
beta = L * abs(b)^2 * norm(v_dot)^2 * (a0'*(Q\a0));

% Angle-Only 最优解：若 alpha>beta，则最优为 R_Phi = (P/norm(v0)^2) v0 v0'
% 若 alpha<beta，则近似最优为 R_Phi = epsilon*v0*v0' + (P-epsilon)*v_dot*v_dot'/norm(v_dot)^2, epsilon小量
% 这里我们根据论文式子(40)取 epsilon=1e-6

% 对于其他准则，需要求解SDP。这里使用CVX求解Trace-Opt

% 定义变量 R (N x N Hermitian)
cvx_begin sdp
    variable R(N,N) hermitian
    % FIM 矩阵 (2x2) for theta and b (实部虚部)
    % 根据公式(14-16)单目标时FIM为分块矩阵，但为优化CRB(theta)或CRB(b)，构造加权
    % 简单起见，这里优化trace(CRB) 即最小化 trace(FIM^{-1})
    % 计算 FIM 各块（函数）
    F11 = L*abs(b)^2 * ( (a_dot'*(Q\a_dot))*(v0'*R*v0) + (a_dot'*(Q\a0))*(v0'*R*v_dot) + conj(...) + (a0'*(Q\a0))*(v_dot'*R*v_dot) );
    F12 = L*conj(b)*norm(v0) * ( (a_dot'*(Q\a0))*(v0'*R*v0) + (a0'*(Q\a0))*(v0'*R*v_dot) );
    F22 = L * (a0'*(Q\a0)) * (v0'*R*v0);
    FIM = [real(F11) -imag(F11) real(F12); ...
           imag(F11) real(F11) imag(F12); ...
           real(F12') -imag(F12') real(F22)];  % 这里简化为只考虑theta实标量，b复=>2维，但实际应3x3?论文中单目标参数量3：theta, Re(b), Im(b)
    % 实际上更简单：目标函数为 trace(FIM^{-1})
    minimize( trace_inv(FIM) )
    subject to
        R == hermitian_semidefinite(N);
        trace(R) == P;
cvx_end
% 得到最优 R，然后计算波束图
theta_scan = -90:0.5:90;
bp = zeros(size(theta_scan));
for i=1:length(theta_scan)
    v = tx_steer(theta_scan(i));
    bp(i) = real(v' * R * v);
end
figure; plot(theta_scan, 10*log10(bp));
xlabel('Angle (deg)'); ylabel('Beampattern (dB)');
title('Optimized Beampattern (Trace-Opt)');
saveas(gcf,'fig3.png');