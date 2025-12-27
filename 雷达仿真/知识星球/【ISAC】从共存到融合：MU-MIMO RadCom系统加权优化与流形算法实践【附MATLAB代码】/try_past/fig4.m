% Fig.4(a): Sidelobe Suppression vs SINR threshold (Gamma)

clear; clc;
% ================== 基础参数 ======================
Nc = 16; Nr = 16;
fc = 5e9; c = 3e8; lamb = c/fc;
spacing = lamb/2;
radAntLoc = spacing*(0:Nr-1);
commAntLoc = spacing*(0:Nc-1);
Pc = 5; Pr = 5; N0 = 1;

angleSpace = linspace(-pi/2, pi/2, 181);
angleSpaceDeg = linspace(-90, 90, 181);

% 雷达目标方向
radAngles = [25 50];
K = length(radAngles);

% 通信用户方向（论文中选定为两个）
commAngles = [90 -15]; 
N = length(commAngles);

% beamwidth 用于划定主瓣
beamwidth = 5;

% 生成 steering vectors
a1 = zeros(Nr, length(angleSpace));
a2 = zeros(Nc, length(angleSpace));
for j = 1:Nr
    a1(j,:) = exp(1i*2*pi*radAntLoc(j)/lamb * sin(angleSpace));
end
for j = 1:Nc
    a2(j,:) = exp(1i*2*pi*commAntLoc(j)/lamb * sin(angleSpace));
end
a = [a1; a2];

% 雷达目标 steering
aRad = zeros(Nr,K);
for k = 1:K
    aRad(:,k) = a1(:,90+radAngles(k));
end

% 通信 steering
g = zeros(Nc,N); f = zeros(Nr,N);
for n = 1:N
    g(:,n) = a2(:,90+commAngles(n));
    f(:,n) = a1(:,90+commAngles(n));
end

% 构造旁瓣区域掩码
isSidelobeRegion = zeros(length(angleSpace),1);
for idx = 1:length(angleSpaceDeg)
    if min(abs(angleSpaceDeg(idx) - radAngles(:))) >= beamwidth
        isSidelobeRegion(idx) = 1;
    end
end

% ========== 仿真主循环：不同 Gamma ============
Gamma_list = linspace(0.01, 1, 10);
sidelobe_sep = zeros(size(Gamma_list));
sidelobe_shared = zeros(size(Gamma_list));
sidelobe_ideal = zeros(size(Gamma_list));

for g_idx = 1:length(Gamma_list)
    Gamma = Gamma_list(g_idx) * [1 1];  % 双用户相同SINR门限

    %% 雷达方向图优化（固定 R1）
    cvx_begin quiet
        variable R1(Nr,Nr) hermitian
        variable t
        maximize( t )
        subject to
            for idx = 1:181
                if isSidelobeRegion(idx)
                    for k = 1:K
                        aMain = a1(:,90+radAngles(k));
                        aSl = a1(:,idx);
                        (aMain'*R1*aMain) - (aSl'*R1*aSl) >= t;
                    end
                end
            end
            for k = 1:K
                am = a1(:,90+radAngles(k));
                aL = a1(:,90+radAngles(k)-beamwidth);
                aR = a1(:,90+radAngles(k)+beamwidth);
                aL'*R1*aL == 0.5 * (am'*R1*am);
                aR'*R1*aR == 0.5 * (am'*R1*am);
            end
            diag(R1) == Pr/Nr * ones(Nr,1);
            R1 == hermitian_semidefinite(Nr);
    cvx_end

    %% 通信联合波束形成（Separated Deployment）
    cvx_begin quiet
        variable W1(Nc,Nc) hermitian
        variable W2(Nc,Nc) hermitian
        variable sigma_temp
        minimize( square_pos(norm( diag( a2'*(W1+W2)*a2 - sigma_temp*(a1'*R1*a1) ), 2)) );
        subject to
            real(trace(g(:,1)*g(:,1)'*W1) - Gamma(1)*( trace(g(:,2)*g(:,2)'*W1) + trace(f(:,1)*f(:,1)'*R1) )) >= N0*Gamma(1);
            real(trace(g(:,2)*g(:,2)'*W2) - Gamma(2)*( trace(g(:,1)*g(:,1)'*W2) + trace(f(:,2)*f(:,2)'*R1) )) >= N0*Gamma(2);
            trace(W1 + W2) <= Pc;
            sigma_temp >= 0;
            W1 == hermitian_semidefinite(Nc);
            W2 == hermitian_semidefinite(Nc);
    cvx_end
    Wsum = W1 + W2;

    %% 方向图计算
    C_sep = blkdiag(R1, Wsum);   % Separated
    C_shared = blkdiag(zeros(Nr), Wsum) + blkdiag(R1, zeros(Nc));  % Shared
    C_ideal = blkdiag(R1, zeros(Nc));  % Ideal Radar-only

    Pd_sep = zeros(1,181);
    Pd_shared = zeros(1,181);
    Pd_ideal = zeros(1,181);

    for i = 1:181
        Pd_sep(i) = abs(a(:,i)' * C_sep * a(:,i));
        Pd_shared(i) = abs(a(:,i)' * C_shared * a(:,i));
        Pd_ideal(i) = abs(a(:,i)' * C_ideal * a(:,i));
    end

    sidelobe_sep(g_idx) = max(Pd_sep(isSidelobeRegion==1));
    sidelobe_shared(g_idx) = max(Pd_shared(isSidelobeRegion==1));
    sidelobe_ideal(g_idx) = max(Pd_ideal(isSidelobeRegion==1));
end

%% =================== 绘图 ===================
figure;
plot(Gamma_list, sidelobe_sep, 'r-o', 'LineWidth', 2); hold on;
plot(Gamma_list, sidelobe_shared, 'b--s', 'LineWidth', 2);
plot(Gamma_list, sidelobe_ideal, 'k:', 'LineWidth', 2);
xlabel('\Gamma (User SINR Requirement)');
ylabel('Sidelobe Level');
legend('Separated','Shared','Radar-only');
title('Fig.4(a): Sidelobe Level vs SINR Constraint');
grid on;
