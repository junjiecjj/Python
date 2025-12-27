%% ==========================================================
%  Fig. 5 replication - PSLR vs SINR (Γ)
%  Deployment: Separated
%  Paper: "MU-MIMO Communications with MIMO Radar"
%  Fan Liu et al., IEEE TWC 2018
% ==========================================================
clc; clear; close all;
cvx_quiet(true);

%% 基本参数设置
Nr = 16; Nc = 16;                % 雷达/通信阵元数
fc = 5e9; c = 3e8; lambda = c/fc; d = lambda/2;
angleSpaceDeg = -90:1:90;
angleSpace = deg2rad(angleSpaceDeg);

radAngles = [-60 -30 0 30 60];    % 雷达目标方向
beamwidth = 10;                   % 主瓣角度范围
commAngles = [-45 -15 15 45];     % 通信用户方向
N = length(commAngles);

Pc = 1; Pr = 1; N0 = 1;           % 通信功率、雷达功率、噪声
Gamma_dB = 4:2:14; Gamma = 10.^(Gamma_dB/10);
M = length(angleSpaceDeg);        % 角度采样点数

%% 生成方向向量矩阵
a1 = zeros(Nr, M); a2 = zeros(Nc, M);
for j = 1:Nr
    a1(j,:) = exp(1i * 2*pi * d*(j-1)/lambda .* sin(angleSpace));
end
for j = 1:Nc
    a2(j,:) = exp(1i * 2*pi * d*(j-1)/lambda .* sin(angleSpace));
end

% 通信通道方向向量
g = zeros(Nc, N); f = zeros(Nr, N);
for n = 1:N
    g(:,n) = a2(:,90+commAngles(n));
    f(:,n) = a1(:,90+commAngles(n));
end

%% 定义 PSLR 计算函数
calcPSLR = @(Ppat, radAngles, beamwidth) ...
    10*log10(...
        max(arrayfun(@(x) max(Ppat(abs(angleSpaceDeg - x) <= beamwidth/2)), radAngles)) / ...
        max(Ppat(~ismember(1:length(angleSpaceDeg), ...
        find(any(abs(angleSpaceDeg - radAngles') <= beamwidth/2,1))))));

%% 初始化结果存储
PSLR_sep_radonly = zeros(size(Gamma));
PSLR_sep_radcom = zeros(size(Gamma));

%% 主循环：扫描 Γ 值
for idx = 1:length(Gamma)
    fprintf('Running Separated Deployment for Γ = %.1f dB\n', Gamma_dB(idx));

    %% === Step 1: 仅雷达优化 Radar-only（式12） ===
    cvx_begin sdp quiet
        variable R1(Nr,Nr) hermitian semidefinite
        variable alpha_val
        minimize( sum_square_abs(alpha_val - diag(a1'*R1*a1)) )
        subject to
            diag(R1) == Pr/Nr;
    cvx_end

    % 计算雷达-only 的波束图
    Prad = abs(diag(a1'*R1*a1));
    PSLR_sep_radonly(idx) = calcPSLR(Prad, radAngles, beamwidth);

    %% === Step 2: 联合雷达通信优化 Joint RadCom（式19） ===
    cvx_begin sdp quiet
        variable Wsum(Nc,Nc) hermitian semidefinite
        variable sigma_temp
        minimize( norm( diag(a2'*Wsum*a2 - sigma_temp*(a1'*R1*a1)) , 2) )
        subject to
            % SINR 约束
            for n = 1:N
                real(trace(g(:,n)*g(:,n)'*Wsum) ...
                    - Gamma(idx)*(trace(f(:,n)*f(:,n)'*R1) + N0)) >= 0;
            end
            trace(Wsum) <= Pc;
            sigma_temp >= 0;
    cvx_end

    % 计算联合波束图
    a = [a1; a2];
    C = blkdiag(R1, Wsum);
    Pd = abs(diag(a'*C*a));
    PSLR_sep_radcom(idx) = calcPSLR(Pd, radAngles, beamwidth);
end

%% 绘图
figure; hold on;
plot(Gamma_dB, PSLR_sep_radcom, 'b-s', 'LineWidth', 2, 'MarkerSize', 8);
plot(Gamma_dB, PSLR_sep_radonly, 'b-^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('\Gamma (dB)');
ylabel('PSLR (dB)');
title('K = 4');
legend('Separated, RadCom', 'Separated, Radar-Only');
grid on; ylim([4 18]); xlim([min(Gamma_dB) max(Gamma_dB)]);
set(gca,'FontSize',12);
