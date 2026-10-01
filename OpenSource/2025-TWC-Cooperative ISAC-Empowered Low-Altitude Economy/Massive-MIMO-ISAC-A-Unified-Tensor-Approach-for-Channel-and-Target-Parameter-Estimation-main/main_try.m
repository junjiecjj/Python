% 清除环境
clc; clear; close all;
    %% 1. 系统参数设置 (对齐原论文)
    fc = 4.9e9; c0 = 3e8; lambda = c0/fc;
    BW = 20e6; M = 300; N = 7; delta_f = 30e3; Ts = 35.677e-6;
    P = 16; Q = 24; L = P * Q; R = 64; 

    % 横轴：发射功率 (dBm)
    P_tx_dBm = 45:5:65; 
    MC_trials = 100; % 为了曲线平滑，建议最终跑图时设为 100+
    K_list = [2, 4];

    % 初始化结果存储 (2种K模式 x 功率点)
    RES = struct('aoa', zeros(2, length(P_tx_dBm)), ...
                 'range', zeros(2, length(P_tx_dBm)), ...
                 'vel', zeros(2, length(P_tx_dBm)), ...
                 'pos', zeros(2, length(P_tx_dBm)));

    %% 2. 仿真核心循环
    for k_idx = 1:length(K_list)
        K = K_list(k_idx);
        % 设置目标真值
        d_true = [180, 30, 100, 150]; d_true = d_true(1:K);          
        v_true = [10, -20, 5, -10]; v_true = v_true(1:K);            
        th_true = deg2rad([10, 25, 40, 55]); th_true = th_true(1:K); 
        ph_true = deg2rad([30, 60, 15, 45]); ph_true = ph_true(1:K);
        
        for p = 1:length(P_tx_dBm)
            curr_P = P_tx_dBm(p);
            fprintf('Calculating K=%d, Power=%d dBm ', K, curr_P);
            err_sq = zeros(1, 4); % 累加器: [range, vel, aoa, pos]
            
            for mc = 1:MC_trials
                % A. 信号生成 (SNR随功率线性变化)
                snr = curr_P - 45; 
                [Y, Frx_c] = gen_sig(K, d_true, v_true, th_true, ph_true, P, Q, R, L, M, N, delta_f, Ts, lambda, snr, mc);

                % B. 运行核心创新算法: Spatial Smoothing CPD
                [A1, A2, ~, z_hat] = Spatial_Smoothing_CPD_Internal(Y, K);

                % C. 误差评估与配对
                [ed, ev, ea, ep] = run_proposed_eval(z_hat, A2, A1, Frx_c, K, P, Q, d_true, v_true, th_true, ph_true, delta_f, c0, Ts, lambda);
                err_sq = err_sq + [ed, ev, ea, ep];
                
                if mod(mc, 10) == 0, fprintf('.'); end
            end
            
            % 记录 RMSE
            RES.range(k_idx, p) = sqrt(err_sq(1) / (MC_trials * K));
            RES.vel(k_idx, p)   = sqrt(err_sq(2) / (MC_trials * K));
            RES.aoa(k_idx, p)   = sqrt(err_sq(3) / (MC_trials * K));
            RES.pos(k_idx, p)   = sqrt(err_sq(4) / (MC_trials * K));
            fprintf(' Done\n');
        end
    end

    %% 3. 绘图 (四宫格展示)
    figure('Color', 'w', 'Position', [100 100 900 700]);
    titles = {'(a) AoA', '(b) Range', '(c) Radial velocity', '(d) Position'};
    y_labels = {'RMSE (deg)', 'RMSE (m)', 'RMSE (m/s)', 'RMSE (m)'};
    fields = {'aoa', 'range', 'vel', 'pos'};

    for i = 1:4
        subplot(2, 2, i);
        % K=2 实线，K=4 虚线
        semilogy(P_tx_dBm, RES.(fields{i})(1,:), 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
        semilogy(P_tx_dBm, RES.(fields{i})(2,:), 'b--s', 'LineWidth', 1.8, 'MarkerSize', 7);
        grid on; xlabel('Transmit power (dBm)'); ylabel(y_labels{i}); title(titles{i});
        if i == 1, legend('Proposed, K=2', 'Proposed, K=4', 'Location', 'best'); end
    end


%% --- 核心子函数：空间平滑 CPD ---
function [A1, A2, A3, z_hat] = Spatial_Smoothing_CPD_Internal(Y, K)
    [R, N, M] = size(Y);
    % 张量 Mode-3 展开并转置
    Y1_T = reshape(Y, R, N*M).'; 
    % 确定子阵参数
    L1 = floor(M/2); L2 = M + 1 - L1; 
    % 预分配内存构建平滑矩阵 Ys
    Ys = zeros(L1*N, R*L2); 
    for l = 1:L2
        Ys(:, (l-1)*R + 1 : l*R) = Y1_T((l-1)*N + 1 : (l+L1-1)*N, :);
    end
    % SVD 降维
    [U, S, V] = svds(Ys, K); 
    % TLS-ESPRIT 提取特征值
    Xi = pinv(U(1:(L1-1)*N, :)) * U(N+1:L1*N, :); 
    [M_mat, Z_diag] = eig(Xi); 
    z_hat = diag(Z_diag);
    % 重构因子矩阵
    A3 = (z_hat.^(0:M-1)).'; 
    P_mat = inv(M_mat).';
    A2 = zeros(N, K); A1 = zeros(R, K);
    for k = 1:K
        A2(:, k) = (kron(A3(1:L1, k)', eye(N)) * U * M_mat(:, k)) / (A3(1:L1, k)' * A3(1:L1, k));
        A1(:, k) = (kron(A3(1:L2, k)', eye(R)) * conj(V) * S * P_mat(:, k)) / (A3(1:L2, k)' * A3(1:L2, k));
    end
end

%% --- 辅助函数：信号生成 ---
function [Y, Frx_c] = gen_sig(K, d, v, th, ph, P, Q, R, L, M, N, df, Ts, lam, snr, mc)
    Yc = zeros(R, N, M); Frx_c = cell(1, K);
    for k = 1:K
        vt = sin(th(k))*cos(ph(k)); ps = cos(th(k));
        a_upa = kron(exp(1j*pi*(0:Q-1)'*ps), exp(1j*pi*(0:P-1)'*vt));
        rng(mc+k*100); 
        Frx = (randn(L,R)+1j*randn(L,R))/sqrt(2*R); Frx_c{k} = Frx;
        bk = Frx'*a_upa*(a_upa' * (randn(L,1)+1j*randn(L,1))/sqrt(2));
        ok = exp(1j*2*pi*Ts*(2*v(k)/lam)*(0:N-1)');
        gk = exp(-1j*2*pi*df*(2*d(k)/3e8)*(0:M-1)');
        Yc = Yc + (randn+1j*randn) * reshape(kron(gk, kron(ok, bk)), [R, N, M]);
    end
    sig_p = norm(Yc(:))^2/numel(Yc);
    Y = Yc + sqrt(sig_p*10^(-snr/10))*(randn(size(Yc))+1j*randn(size(Yc)))/sqrt(2);
end

%% --- 辅助函数：误差评估 ---
function [ed, ev, ea, ep] = run_proposed_eval(z, A2, A1, Frx, K, P, Q, dt, vt, tht, pht, df, c0, Ts, lam)
    ed=0; ev=0; ea=0; ep=0; rem=1:K;
    for k = 1:K
        % 1. 距离配对
        cur_dists = abs(angle(z(rem))/(-2*pi*df)*c0/2);
        [~, m] = min(abs(cur_dists - dt(k))); midx = rem(m);
        
        dk_e = abs(angle(z(midx))/(-2*pi*df)*c0/2);
        vk_e = (angle(A2(2,midx)/A2(1,midx))/(2*pi*Ts))*lam/2;
        % 调用真实 AoA 搜索 (确保此函数在路径下)
        [th_e, ph_e] = GRQ_AoA_Method(A1(:, midx), Frx{k}, P, Q); 
        
        ed = ed + (dk_e - dt(k))^2;
        ev = ev + (vk_e - vt(k))^2;
        ea = ea + (rad2deg(th_e) - rad2deg(tht(k)))^2;
        % Position RMSE 计算
        ep = ep + (dk_e - dt(k))^2 + (dt(k) * (th_e - tht(k)))^2;
        
        rem(m) = [];
    end
end