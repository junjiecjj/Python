clc; clear; close all;

%% 参数设置
M = 64;                 % 天线总数
N = 4;                  % 通信用户数
P = 1;                  % 发射总功率
lambda = 1;             % 波长
d = lambda/2;           % 阵元间距
angleSpaceDeg = -90:0.1:90;
angleSpaceRad = deg2rad(angleSpaceDeg);

% 雷达目标角度（Fig. 3a中5个主瓣）
radAngles = [-60, -30, 0, 30, 60];
radAnglesRad = deg2rad(radAngles);
beamwidth = 10;  % 主瓣宽度 (degree)

% 通信用户方向（避开目标方向）
commAngles = [-45, -15, 15, 45];
commAnglesRad = deg2rad(commAngles);

%% 构造阵列方向向量
a = @(theta) exp(1j*2*pi*d*(0:M-1)'*sin(theta));

% 构造理想雷达波束图 Pdesired
Pdesired = zeros(size(angleSpaceRad));
for k = 1:length(radAnglesRad)
    mainlobe = abs(angleSpaceDeg - radAngles(k)) <= beamwidth/2;
    Pdesired(mainlobe) = 1;
end

% 归一化
Pdesired = Pdesired / max(Pdesired);

%% 构造雷达协方差矩阵 R1
A_rad = zeros(M, length(radAnglesRad));
for k = 1:length(radAnglesRad)
    A_rad(:, k) = a(radAnglesRad(k));
end

% CVX 设计 R1，使波束主瓣位于目标方向，旁瓣最小化
cvx_begin sdp quiet
    variable R1(M, M) hermitian semidefinite
    expression beampattern(length(angleSpaceRad))
    for i = 1:length(angleSpaceRad)
        beampattern(i) = real(a(angleSpaceRad(i))' * R1 * a(angleSpaceRad(i)));
    end
    minimize( norm(beampattern - Pdesired', 2) )
    subject to
        trace(R1) == P;
cvx_end

% 计算雷达波束图
BPrad = zeros(size(angleSpaceRad));
for i = 1:length(angleSpaceRad)
    BPrad(i) = real(a(angleSpaceRad(i))' * R1 * a(angleSpaceRad(i)));
end
BPrad = BPrad / max(BPrad);  % 归一化

%% 构造通信协方差矩阵 W1 ~ WN
H = zeros(N, M);  % 每一行为一个通信用户方向的 steering vector
for k = 1:N
    H(k,:) = a(commAnglesRad(k))';
end

cvx_begin sdp quiet
    variable W(M, M) hermitian semidefinite
    variable SINR_th
    minimize( norm(real(a(angleSpaceRad)' * (R1 + W) * a(angleSpaceRad))' - Pdesired, 2) )
    subject to
        trace(W) <= P;
        for k = 1:N
            real(H(k,:) * W * H(k,:)') >= SINR_th;  % 最小 SINR 约束
        end
cvx_end

% 通信波束图（仅用于调试）
BPcomm = zeros(size(angleSpaceRad));
for i = 1:length(angleSpaceRad)
    BPcomm(i) = real(a(angleSpaceRad(i))' * W * a(angleSpaceRad(i)));
end
BPcomm = BPcomm / max(BPcomm);

% 联合波束图 Pd
C_total = R1 + W;
Pd = zeros(size(angleSpaceRad));
for i = 1:length(angleSpaceRad)
    Pd(i) = real(a(angleSpaceRad(i))' * C_total * a(angleSpaceRad(i)));
end
Pd = Pd / max(Pd);

%% 绘图 - 完全对齐 Fig. 3(a)
figure;
plot(angleSpaceDeg, Pdesired, 'k--', 'LineWidth', 1.5); hold on;    % Ideal
plot(angleSpaceDeg, BPrad, 'b--', 'LineWidth', 2); hold on;         % Radar Only
plot(angleSpaceDeg, Pd, 'r-', 'LineWidth', 2); hold on;             % RadCom

% 可选：标记通信方向（论文图中未画出）
for n = 1:N
    line([commAngles(n), commAngles(n)], [0, max(Pd)*1.05], 'LineStyle', ':', 'Color', [0.4 0.4 0.4]);
end

xlabel('Angle (Degree)');
ylabel('Normalized Beampattern');
title('Separated Deployment');
legend('Ideal', 'Radar-Only', 'RadCom');
xlim([-90 90]); ylim([0 1.2]);
set(gca, 'FontSize', 12);
grid on;
