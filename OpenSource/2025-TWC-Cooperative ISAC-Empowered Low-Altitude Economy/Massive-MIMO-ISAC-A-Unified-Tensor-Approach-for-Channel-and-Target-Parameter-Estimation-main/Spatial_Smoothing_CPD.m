function [A1, A2, A3, z_hat] = Spatial_Smoothing_CPD(Y, K)
    [R, N, M] = size(Y);
    % 步骤 1: Mode-1 展开 [cite: 1298, 1299]
    Y1 = reshape(Y, R, N*M); 
    Y1_T = Y1.'; 

    % 步骤 2: 空间平滑 [cite: 1300, 1301]
    L1 = floor(M/2); L2 = M + 1 - L1; % 满足平滑条件 [cite: 1300]
    Ys = [];
    for l = 1:L2
        % 选择对应的子块进行堆叠 [cite: 1301]
        block = Y1_T((l-1)*N + 1 : (l+L1-1)*N, :);
        Ys = [Ys, block];
    end

    % 步骤 3: SVD 获取信号子空间 [cite: 1261, 1262]
    [U, S, V] = svds(Ys, K);

    % 步骤 4: 旋转不变性求解生成器 (ESPRIT) [cite: 1285, 1291]
    U1 = U(1:(L1-1)*N, :);
    U2 = U(N+1:L1*N, :);
    Xi = pinv(U1) * U2; 
    [M_mat, Z_diag] = eig(Xi);
    z_hat = diag(Z_diag);

    % 步骤 5-8: 因子矩阵重构 [cite: 1294, 1305, 1309, 1315]
    A3 = zeros(M, K);
    A2 = zeros(N, K);
    A1 = zeros(R, K);
    P_mat = inv(M_mat).';
    
    for k = 1:K
        A3(:, k) = (z_hat(k).^(0:M-1)).'; % 恢复时延矩阵 [cite: 1305]
        % 恢复多普勒矩阵 [cite: 1309]
        A2(:, k) = (kron(A3(1:L1, k)', eye(N)) * U * M_mat(:, k)) / (A3(1:L1, k)' * A3(1:L1, k));
        % 恢复空间矩阵 [cite: 1315]
        A1(:, k) = (kron(A3(1:L2, k)', eye(R)) * conj(V) * S * P_mat(:, k)) / (A3(1:L2, k)' * A3(1:L2, k));
    end
end