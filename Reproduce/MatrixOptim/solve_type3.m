function X_opt = solve_type3(H, S, R_d, L)
    % 求解问题：min ||H X - S||_F^2  s.t. (1/L) X X^H = R_d
    % 输入：H - M×N, S - M×L, R_d - N×N Hermitian 正定, L (要求 L ≥ N)
    % 输出：X_opt = F * V_A * U_{A,1}^H，其中 F 来自 C = L*R_d = F*F^H
    [M, N_h] = size(H);
    N = size(R_d, 1);          % 从 R_d 获取 N
    if N_h ~= N
        error('H 的列数必须与 R_d 的维度一致');
    end
    [M_s, L_s] = size(S);
    if M_s ~= M || L_s ~= L
        error('S 的尺寸必须与 H X 匹配');
    end
    if L < N
        error('约束要求 L ≥ N，否则无解');
    end
    if ~isequal(R_d, R_d')
        error('R_d 必须是 Hermitian 矩阵');
    end
    
    C = L * R_d;
    % Cholesky 分解 C = F * F^H (F 为下三角)
    F = chol(C, 'lower');           % N×N
    H_tilde = H * F;                % M×N
    A = S' * H_tilde;               % L×N
    [U_A, ~, V_A] = svd(A, 'econ'); % U_A: L×L, V_A: N×N (因为 L≥N)
    U_A1 = U_A(:, 1:N);             % L×N
    Z_opt = V_A * U_A1';            % N×L
    X_opt = F * Z_opt;
end