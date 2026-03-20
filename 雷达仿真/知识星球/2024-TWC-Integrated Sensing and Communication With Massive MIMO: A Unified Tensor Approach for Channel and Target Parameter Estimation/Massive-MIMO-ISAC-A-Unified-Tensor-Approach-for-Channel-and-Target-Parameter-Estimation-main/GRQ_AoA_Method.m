function [theta, phi] = GRQ_AoA_Method(b_hat, Frx, P, Q)
    % 步骤 1: 1-D 搜索垂直分量 psi = cos(theta)
    psi_grid = -1:0.001:1; 
    trace_val = zeros(size(psi_grid));
    for i = 1:length(psi_grid)
        psi = psi_grid(i);
        aq = exp(1j*pi*(0:Q-1)'*psi);
        T_mat = kron(aq, eye(P)); 
        Q1 = T_mat' * Frx * (b_hat * b_hat') * Frx' * T_mat;
        Q2 = T_mat' * Frx * Frx' * T_mat + 1e-6*eye(P);
        trace_val(i) = abs(trace(Q2 \ Q1)); % 使用左除优化
    end
    [~, idx] = max(trace_val);
    psi_hat = psi_grid(idx);

    % 步骤 2: 恢复水平分量导向矢量
    aq_hat = exp(1j*pi*(0:Q-1)'*psi_hat);
    T_hat = kron(aq_hat, eye(P));
    Phi = (T_hat' * Frx * Frx' * T_hat + 1e-6*eye(P)) \ (T_hat' * Frx * (b_hat * b_hat') * Frx' * T_hat);
    [V_eig, D_eig] = eig(Phi);
    [~, max_idx] = max(abs(diag(D_eig))); 
    ap_est = V_eig(:, max_idx);
    ap_est = ap_est * exp(-1j * angle(ap_est(1)));

    % 步骤 3: 提取水平分量 vartheta = sin(theta)cos(phi)
    v_grid = -1:0.001:1;
    v_corr = zeros(size(v_grid));
    for j = 1:length(v_grid)
        ap_test = exp(1j * pi * (0:P-1)' * v_grid(j));
        v_corr(j) = abs(ap_test' * ap_est);
    end
    [~, v_idx] = max(v_corr);
    vartheta_hat = v_grid(v_idx);

    % 换算物理角度
    theta = acos(psi_hat);
    % 限制 cos_phi 范围并计算
    denom = sin(theta);
    if denom < 1e-4, denom = 1e-4; end
    cos_phi = vartheta_hat / denom;
    cos_phi = max(min(cos_phi, 1), -1);
    phi = acos(cos_phi);
end