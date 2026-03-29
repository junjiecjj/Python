




function X_opt = solve_type1(S, a)
    % 求解问题：min ||X - S||_F^2  s.t. X X^H = a I
    % 输入：S - M×N 复数矩阵，a > 0（要求 M ≤ n）
    % 输出：X_opt = sqrt(a) * U * V^H，其中 S = U Σ V^H 是奇异值分解
    [M, N] = size(S);
    if M > N
        error('约束 X X^H = a I 在 m > n 时无解，需要 m ≤ n');
    end
    if a <= 0
        error('参数 a 必须为正数');
    end
    
    % 奇异值分解 S = U Σ V^H
    [U, ~, V] = svd(S, 'econ');   % U: m×m, V: n×n (因为 m≤n)
    X_opt = sqrt(a) * (U * V');   % V' 即 V 的共轭转置
end