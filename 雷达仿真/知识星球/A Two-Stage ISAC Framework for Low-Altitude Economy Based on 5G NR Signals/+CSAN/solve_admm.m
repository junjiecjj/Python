function [Upsilon] = solve_admm(r_bar, S_tilde, M, N, lambda, rho, max_iter,tol_abs, tol_rel)
% solve_admm
%
% 使用 ADMM 算法求解问题 (公式 34-36)。
% 使用了我们讨论中修正后的更新规则。
% (e_bar == 0 的简化版本)

    MN = M * N;

    % --- 预计算 ---
    S_tilde_H = S_tilde';
    S_H_S = S_tilde_H * S_tilde;
    % 预计算 z 更新中需要求逆的项
    inv_term = inv(S_H_S + 2*rho*eye(MN));

    % I_center 是一个 (2M-1)x(2N-1) 矩阵，仅在 (M,N) 处为 1
    % 用于 U 更新 (公式 44)
    I_center = zeros(2*M-1, 2*N-1);
    I_center(M, N) = 1;

    % --- 初始化变量 ---
    z_bar = zeros(MN, 1);

    U = zeros(2*M-1, 2*N-1);
    t = 0;

    % ADMM 变量
    Theta = complex(zeros(MN + 1, MN + 1));
    Upsilon = complex(zeros(MN + 1, MN + 1));

    % 存储目标函数历史
    obj_hist = zeros(max_iter, 1);
    fprintf('ADMM 迭代中... (共 %d 次):\n', max_iter);

        % 收敛判断所需的变量
    pri_res_hist = zeros(max_iter, 1);
    dual_res_hist = zeros(max_iter, 1);
    eps_pri_hist = zeros(max_iter, 1);
    eps_dual_hist = zeros(max_iter, 1);

    value=0;


    for iter = 1:max_iter


        pre_value=value;

        if mod(iter, 50) == 0
            fprintf('迭代 %d/%d\n', iter, max_iter);
        end

        % --- 1. 更新 z_bar (公式 42, 移除 e_bar) ---
        Theta_1_vec = Theta(1:MN, MN+1);   % \theta_1^l
        Upsilon_1_vec = Upsilon(1:MN, MN+1); % v_1^l

        % 从 rhs_z 中移除 e_bar
        rhs_z = S_tilde_H * r_bar + 2*rho*Theta_1_vec + 2*Upsilon_1_vec;
        z_bar = inv_term * rhs_z;

 

        % aug_lag = calculate_augmented_lagrangian(r_bar, S_tilde, z_bar, U, t, M, N, lambda, rho, Theta,  Upsilon);

        % --- 3. 更新 t (公式 43) ---
        Theta_t = Theta(MN+1, MN+1);
        Upsilon_t = Upsilon(MN+1, MN+1);
        t = Theta_t + (Upsilon_t - lambda/2) / rho;
        t = real(t); % t 必须是实数

        % aug_lag =calculate_augmented_lagrangian(r_bar, S_tilde, z_bar, U, t, M, N, lambda, rho, Theta,  Upsilon);

        % --- 4. 更新 U (修正后的公式 44) ---
        Theta_0 = Theta(1:MN, 1:MN);
        Upsilon_0 = Upsilon(1:MN, 1:MN);

        P_matrix = Theta_0 + Upsilon_0 / rho;

        % T_adjoint 是 T* 的未归一化版本 (即论文(45)中没有分母)
        T_adj_P = CSAN.T_adjoint(P_matrix, M, N);

        % (N-|j|)(M-|k|) 的计数矩阵
        [J, K] = meshgrid(-N+1:N-1, -M+1:M-1);
        U_counts = (N - abs(J)) .* (M - abs(K));
        % 避免除以零
        

        % 归一化 T*
        T_star_P = T_adj_P ./ U_counts;

        % 更新U
        U = T_star_P - (lambda / (2*M*N*rho)) * I_center;


        % aug_lag = calculate_augmented_lagrangian(r_bar, S_tilde, z_bar, U, t, M, N, lambda, rho, Theta,  Upsilon);

        % --- 5. 构建 M 矩阵 (约束的右侧) ---
        T_U = CSAN.build_T(U, M, N); % 使用更新后的 U
        M_matrix = [T_U, z_bar; z_bar', t]; % 使用更新后的 z, t


        % --- 6. 更新 Theta (公式 49) ---
        Target_matrix = M_matrix - Upsilon / rho;

        % 确保 Hermitian 以避免 eig 出错
        Target_matrix = (Target_matrix + Target_matrix') / 2; 

        [V, D] = eig(double(Target_matrix)); % 强制转换为 double

        D_plus = max(D, 0); % 将负特征值设为 0
        Theta = V * D_plus * V';
        Theta = (Theta + Theta') / 2; % 再次确保

        % aug_lag = calculate_augmented_lagrangian(r_bar, S_tilde, z_bar, U, t, M, N, lambda, rho, Theta, Upsilon)

        % --- 7. 更新 Upsilon (公式 36) ---
        Upsilon = Upsilon + rho * (Theta - M_matrix);

        % --- 记录当前的目标函数值 ---
        obj_hist(iter) = CSAN.calculate_objective(r_bar, S_tilde, z_bar, U, t, M, N, lambda);


        
        value=obj_hist(iter);

        % --- 检查收敛条件 ---
        if (abs((pre_value-value)/(pre_value+eps))<1e-2)
            fprintf('  第 %d 次迭代收敛\n', iter);
            break; % 提前退出循环
        end

    end

    fprintf('ADMM 迭代完成。\n');

    %  % --- 从对偶解中恢复频率信息 ---
    % fprintf('正在从对偶解中恢复频率...\n');
    % % 1. 提取对偶解 nu_hat (根据 Lemma 2)
    % nu_hat_v1 = Upsilon(1:MN, MN+1); % 提取 v1_hat
    % nu_hat = -nu_hat_v1 / 2;         % 计算 nu_hat

    % % 2. 在精细网格上搜索对偶多项式的峰值
    % grid_res = 200; % 搜索网格的精细度
    % phi_search = linspace(0, 1, grid_res); % 多普勒频率的搜索范围
    % psi_search = linspace(0, 1, grid_res); % 延迟的搜索范围
    % Q_mag = zeros(grid_res, grid_res);     % 存储对偶多项式 |Q(phi, psi)| 的幅值
    % 
    % 
    % for i = 1:grid_res % 循环遍历所有可能的 phi (对应矩阵的行)
    %     temp_Q_row = zeros(1, grid_res);
    %     for j = 1:grid_res % 循环遍历所有可能的 psi (对应矩阵的列)
    %         phi = phi_search(i);
    %         psi = psi_search(j);
    % 
    %         % 根据 (phi, psi) 构建一个原子
    %         b_phi = exp(1i * 2 * pi * (0:M-1)' * phi);
    %         g_psi = exp(1i * 2 * pi * (0:N-1)' * psi);
    %         atom = kron(conj(g_psi(:)), b_phi(:));
    % 
    %         % 计算对偶多项式的幅值 |<nu_hat, a(phi, psi)>|
    %         temp_Q_row(j) = abs(nu_hat' * atom);
    %     end
    %     Q_mag(i, :) = temp_Q_row;
    % end
    % 
    % 
    % is_peak = imregionalmax(Q_mag);
    % peak_indices = find(is_peak);
    % peak_vals = Q_mag(peak_indices);
    % [~, sorted_order] = sort(peak_vals, 'descend');
    % 
    % num_peaks_to_find = 1;
    % num_peaks = min(length(sorted_order), num_peaks_to_find);
    % if num_peaks == 0
    %     results.phi_est = [];
    %     results.psi_est = [];
    %     results.alpha_est = [];
    %     return;
    % end
    % top_peak_indices = peak_indices(sorted_order(1:num_peaks));
    % 
    % [phi_idx_top, psi_idx_top] = ind2sub(size(Q_mag), top_peak_indices);
    % results.phi_est = phi_search(phi_idx_top)';
    % results.psi_est = psi_search(psi_idx_top)';
    % 
    % % 1. 根据估计出的频率构建响应矩阵 B_est 和 G_est
    % B_est = exp(1i * 2 * pi * (0:M-1)' * results.phi_est');
    % G_est = exp(1i * 2 * pi * (0:N-1)' * results.psi_est');
    % 
    % % 2. 构建包含 K 个原子的小字典矩阵 C_est
    % C_est = zeros(MN, num_peaks);
    % for k=1:num_peaks
    %      C_est(:,k) = kron(conj(G_est(:,k)), B_est(:,k));
    % end
    % 
    % % 3. 求解最小二乘问题: alpha = argmin ||r_vec - S_tilde * C_est * alpha||^2
    % A = S_tilde * C_est;
    % results.alpha_est = (A'*A) \ (A'*r_bar); % 高效的最小二乘解法




end





