



% Sec.III-C. Beampattern Matching Design in "2007-TSP-(Petre Stoica)-On Probing Signal Design For MIMO Radar" 
function [R_opt, alpha, r_opt] = BeampatternMatchingDesign(c, M, w_l, w_c, theta_est, theta_grid, P_des)
    
    K = length(theta_est);               % 目标个数
    L = length(theta_grid);
    % 导向矢量函数（均匀线阵，半波长间距）
    a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1
    %% 2. 构建从R到实向量r的线性映射（这里只需知道r的长度）
    % r 包含：M个对角元 + 2 * (M*(M-1)/2) 个上三角实部虚部
    n_r = M^2;   % 实向量 r 的长度
    %% 3. 构建系数向量 g_l 和 d_{k,p}（数值计算，不涉及CVX变量）
    % 先计算每个网格点的导向矢量
    a_grid = zeros(M, L);
    for l = 1:L
        a_grid(:,l) = a(theta_grid(l));
    end
    % 构建 g_l 矩阵（L行，每行是 1×n_r 的系数向量）
    g = zeros(M^2, L);
    for l = 1:L
        al = a_grid(:,l);
        for j = 1:M^2
            r_unit = zeros(M^2, 1);
            r_unit(j) = 1;
            Rj = r2R(r_unit, M);
            val = al' * Rj * al;
            g(j, l) = -real(val);
        end
    end
    % 构建 d_{k,p} 矩阵 (K×K 元胞，存储每个对的系数向量)
    d = cell(K, K);
    for k = 1:K-1
        ak = a(theta_est(k));
        for p = k+1:K
            ap = a(theta_est(p));
            d_kp = zeros(M^2, 1);
            for j = 1:M^2
                r_unit = zeros(M^2, 1);
                r_unit(j) = 1;
                Rj = r2R(r_unit, M);
                d_kp(j) = ak' * Rj * ap;
            end
            d{k,p} = d_kp;
        end
    end
    %% 4. 构建二次型矩阵 Gamma（数值矩阵）
    % 第一项：来自 beampattern 匹配
    Q1 = zeros(M^2+1, M^2+1);
    for l = 1:L
        x = [P_des(l); g(:,l)];
        Q1 = Q1 + w_l(l) * (x * x.');
    end
    Q1 = Q1 / L;
    % 第二项：来自交叉项抑制
    Q2 = zeros(M^2+1, M^2+1);
    n_pairs = 0;
    for k = 1:K
        for p = k+1:K
            n_pairs = n_pairs + 1;
            x = [0; d{k,p}];
            Q2 = Q2 + x * x.';
        end
    end
    if n_pairs > 0
        Q2 = (2 * w_c / (K^2 - K)) * real(Q2);
    end
    fprintf('norm(Q1) = %e\n', norm(Q1));
    fprintf('norm(Q2) = %e\n', norm(Q2));
    fprintf('norm(Q2)/norm(Q1) = %e\n', norm(Q2)/norm(Q1));
    Gamma = Q1 + Q2;
    % Gamma = (Gamma + Gamma')/2;
    % 计算 Gamma 的平方根
    % [V, D] = eig(Gamma);
    % D = max(D, 0);
    % sqrt_Gamma =  V * sqrt(D) * V';
    % sqrt_Gamma = sqrtm(Gamma);
    sqrt_Gamma = Gamma^(0.5);

    %% 5. 求解 SOCP（使用 CVX）
    % 定义变量：R (Hermitian), r (实向量), alpha_, delta
    cvx_begin quiet sdp
        variable R(M,M) hermitian
        variable r(n_r)
        variable alpha1
        variable delta1 
        % 构建 rho = [alpha_; r]
        rho = [alpha1; r];
        % 目标：min delta
        minimize delta1;
        subject to
            % SOCP 约束
            norm( sqrt_Gamma * rho ) <= delta1;
    
            % 将 r 与 R 的线性关系约束
            idx = 1;
            % 对角线
            for i = 1:M
                R(i,i) == r(idx);
                idx = idx + 1;
            end
            % 上三角的实部虚部
            for i = 1:M-1
                for j = i+1:M
                    R(i,j) == r(idx) + sqrt(-1) * r(idx+1);
                    R(j,i) == conj(R(i,j));  % 利用共轭对称填充下三角
                    %real(R(i,j)) == r(idx);
                    %imag(R(i,j)) == r(idx+1);
                    idx = idx + 2;
                end
            end
            % 对角元固定
            for i = 1:M
                R(i,i) == c / M;
            end
            % 半正定约束
            R == hermitian_semidefinite(M);
    cvx_end
    
    % 6. 输出结果
    if strcmp(cvx_status, 'Solved')
        fprintf('求解成功，最优目标值 delta = %f, 最优 alpha_ = %f\n', delta1, alpha1);
        R_opt = R;  % 直接使用 R
        r_opt = r;
        alpha = alpha1;
        % disp('最优协方差矩阵 R:');
        % disp(R_opt);
    else
        fprintf('求解失败: %s\n', cvx_status);
    end

end
