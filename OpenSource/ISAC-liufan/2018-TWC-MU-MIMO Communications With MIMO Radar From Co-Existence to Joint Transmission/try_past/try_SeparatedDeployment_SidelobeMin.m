% Separated Deployment Beamforming Simulation (Liu et al. 2018)
% By Nate Raymondi, modified by Lucia's AI assistant

% ========== 系统参数 ==========
Nc = 16; Nr = 16;           % 通信 / 雷达发射天线数
fc = 5e9; c = 3e8;
lamb = c/fc; spacing = lamb/2;
radAntLoc = spacing*(0:Nr-1);
commAntLoc = spacing*(0:Nc-1);
Pc = 5; Pr = 5;             % 通信/雷达功率预算
N0 = 1;                     % 噪声方差

% ========== 天线方向向量 ==========
M = 181;
angleSpace = linspace(-pi/2, pi/2, M);
angleSpaceDeg = linspace(-90, 90, M);

a1 = zeros(Nr, M);          % 雷达方向向量矩阵
for j = 1:Nr
    a1(j,:) = exp(1i*2*pi*radAntLoc(j)/lamb .* sin(angleSpace));
end

a2 = zeros(Nc, M);          % 通信方向向量矩阵
for j = 1:Nc
    a2(j,:) = exp(1i*2*pi*commAntLoc(j)/lamb .* sin(angleSpace));
end

a = [a1; a2];               % 总方向向量 (用于整体波束图)

% ========== 雷达/通信目标设定 ==========
K = 2;
radAngles = [25 50];        % 目标角度 (度)
commAngles = [90 -15];      % 用户角度 (度)
Gamma = Pc * [0.5 0.2];     % SINR门限

g = zeros(Nc,2); f = zeros(Nr,2);
for n = 1:2
    g(:,n) = a2(:,90 + commAngles(n));
    f(:,n) = a1(:,90 + commAngles(n));
end

% ========== 设定理想波束图 ==========
Pdesired = zeros(M,1);
beamwidth = 5;
for i = 1:M
    if min(abs(angleSpaceDeg(i) - radAngles(:))) <= beamwidth/2
        Pdesired(i) = 10;  % 主瓣峰值设为10
    else
        Pdesired(i) = 0;
    end
end

% ========== 求解 Eq.12 雷达优化问题 ==========
isSidelobeRegion = zeros(M,1);
for i = 1:M
    if min(abs(angleSpaceDeg(i)-radAngles(:))) >= beamwidth
        isSidelobeRegion(i) = 1;
    end
end

cvx_begin quiet
    variable R1(Nr,Nr) hermitian
    variable t
    minimize( -t )
    subject to
        for i = 1:M
            if isSidelobeRegion(i) == 1
                for k = 1:K
                    a_main = a1(:,90+radAngles(k));
                    a_side = a1(:,i);
                    (a_main'*R1*a_main) - (a_side'*R1*a_side) >= t;
                end
            end
        end
        for k = 1:K
            a_c = a1(:,90+radAngles(k));
            a_l = a1(:,90+radAngles(k)-beamwidth);
            a_r = a1(:,90+radAngles(k)+beamwidth);
            (a_l'*R1*a_l) == (a_c'*R1*a_c)/2;
            (a_r'*R1*a_r) == (a_c'*R1*a_c)/2;
        end
        diag(R1) == Pr/Nr * ones(Nr,1);
        R1 == semidefinite(Nr);
cvx_end

% ========== 求解 Eq.19 通信优化问题 ==========
cvx_begin quiet
    variable W1opt(Nc,Nc) hermitian
    variable W2opt(Nc,Nc) hermitian
    variable sigma_temp
    minimize( square_pos(norm( diag( a2'*(W1opt+W2opt)*a2 - sigma_temp*(a1'*R1*a1) ) ,2)) )
    subject to
        real(trace(g(:,1)*g(:,1)'*W1opt) - Gamma(1)*( trace(g(:,2)*g(:,2)'*W1opt) + trace(f(:,1)*f(:,1)'*R1) )) >= N0*Gamma(1);
        real(trace(g(:,2)*g(:,2)'*W2opt) - Gamma(2)*( trace(g(:,1)*g(:,1)'*W2opt) + trace(f(:,2)*f(:,2)'*R1) )) >= N0*Gamma(2);
        trace(W1opt + W2opt) <= Pc;
        sigma_temp >= 0;
        W1opt == semidefinite(Nc);
        W2opt == semidefinite(Nc);
cvx_end

% ========== 随机化恢复波束向量 ==========
nRand = 10;
for L = 1:nRand
    w1(:,L) = mvnrnd(zeros(Nc,1), W1opt).' + 1i*mvnrnd(zeros(Nc,1), W1opt).';
    w2(:,L) = mvnrnd(zeros(Nc,1), W2opt).' + 1i*mvnrnd(zeros(Nc,1), W2opt).';
    w1(:,L) = sqrt(trace(W1opt)) * w1(:,L) / norm(w1(:,L));
    w2(:,L) = sqrt(trace(W2opt)) * w2(:,L) / norm(w2(:,L));
    W1cov(:,:,L) = w1(:,L)*w1(:,L)';
    W2cov(:,:,L) = w2(:,L)*w2(:,L)';
end

% ========== 筛选满足SINR的beamforming向量 ==========
index = 1;
for i = 1:nRand
    for j = 1:nRand
        if real(trace(g(:,1)*g(:,1)'*W1cov(:,:,i)) - Gamma(1)*(trace(g(:,2)*g(:,2)'*W1cov(:,:,i)) + trace(f(:,1)*f(:,1)'*R1))) >= N0*Gamma(1)
            if real(trace(g(:,2)*g(:,2)'*W2cov(:,:,j)) - Gamma(2)*(trace(g(:,1)*g(:,1)'*W2cov(:,:,j)) + trace(f(:,2)*f(:,2)'*R1))) >= N0*Gamma(2)
                u2F(:,1,index) = w1(:,i);
                u2F(:,2,index) = w2(:,j);
                index = index + 1;
            end
        end
    end
end

% ========== 找到最优对（目标最小） ==========
numFeasible = size(u2F,3);
for i = 1:numFeasible
    Wsum = u2F(:,1,i)*u2F(:,1,i)' + u2F(:,2,i)*u2F(:,2,i)';
    obj(i) = norm( diag( a2'*Wsum*a2 - sigma_temp*a1'*R1*a1 ) )^2;
end
[~,best] = min(obj);
w1 = u2F(:,1,best); W1 = w1*w1';
w2 = u2F(:,2,best); W2 = w2*w2';
Wsum = W1 + W2;

% ========== 波束图绘制 ==========
C = blkdiag(R1, Wsum);
for i = 1:M
    Pd(i) = abs(a(:,i)'*C*a(:,i));
    BPrad(i) = abs(a1(:,i)'*R1*a1(:,i));
    BPcomm(i) = abs(a2(:,i)'*Wsum*a2(:,i));
end

% ========== 绘图：波束图简化风格 ==========
figure;

% 理想波束图（灰色点线）
plot(angleSpaceDeg, Pdesired, ':', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5); hold on;

% 实际联合波束图（黑色粗线）
plot(angleSpaceDeg, Pd, '-', 'Color', [0 0 0], 'LineWidth', 2);

% 雷达部分（深蓝实线）
plot(angleSpaceDeg, BPrad, '-', 'Color', [0.1 0.2 0.8], 'LineWidth', 1.5);

% 通信部分（橙色短点线）
plot(angleSpaceDeg, BPcomm, ':', 'Color', [0.85 0.4 0.1], 'LineWidth', 1.8);

% 用户方向（红色竖线）
for n = 1:2
    xline(commAngles(n), '-', 'Color', [0.8 0 0], 'LineWidth', 1);
end

xlabel('Angle (°)');
ylabel('Beampattern Gain');
title('Beampattern (Separated Deployment)');
legend('Ideal', 'Total', 'Radar Only', 'Comm Only', 'User Angles');
grid on;
xlim([-90 90]);
