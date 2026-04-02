

function beta_AML = AML_estimator(X, S, theta_est)
% AML_estimator 实现公式 (20) 的 AML 幅度估计
% 输入：
%   X         - 接收数据矩阵，维度 M × L
%   S         - 发射波形矩阵，维度 N × L（通常 N = M）
%   theta_est - 估计的目标方向（度），列向量，长度 K
%   R_SS      - 发射波形协方差矩阵，维度 N × N（可选，若不提供则从 S 计算）
% 输出：
%   beta_AML  - 估计的复幅度，列向量，长度 K

    [M, L] = size(X);
    [N, ~] = size(S);
    K = length(theta_est);
    
    % 若未提供 R_SS，则计算样本协方差
    Rss = (S * S') / L;

    % 导向矢量函数（半波长间距均匀线阵）
    at_func = @(theta) exp(1j * pi * (0:N-1)' * sind(theta));
    ar_func = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));
    
    % 构建接收导向矩阵 A_r (M × K)
    A_r = zeros(M, K);
    for k = 1:K
        A_r(:, k) = ar_func(theta_est(k));
    end
    % 构建接收导向矩阵 A_t (N × K)
    A_t = zeros(N, K);
    for k = 1:K
        A_t(:, k) = at_func(theta_est(k));
    end

    % 样本协方差矩阵 R_hat
    Ryy = (X * X') / L;
    % 计算 B = A_r^T * R_SS * A_r^*  (K × K)
    B = A_t.' * Rss * conj(A_t);
    temp1 = X * S' * conj(A_t);      % M × K
    temp2 = A_t.' * S * X';          % K × M
    T = L * Ryy - (1/L) * (temp1 / B * temp2);
    
    % 计算 C = A_r^H * inv(T) * A_r  (K × K)
    C = A_r' / T * A_r;
    
    % 计算 D = A_r^H * R_SS * A_r (K × K)
    D = A_t' * conj(Rss) * A_t;
    
    % Hadamard 积 H = C ⊙ D
    H = C .* D;
    
    % 计算 E = A_r^H * inv(T) * X * S^H * conj(A_r)  (K × K)
    E = A_r' / T * X * S' * conj(A_t);
    vecd_E = diag(E);   % 提取对角元素
    
    % 最终 AML 估计
    beta_AML = (1/L) * (H \ vecd_E);
end