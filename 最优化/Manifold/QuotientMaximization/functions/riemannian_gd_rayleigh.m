function [x, lambda_est, info] = riemannian_gd_rayleigh(A, opts)
    %RIEMANNIAN_GD_RAYLEIGH  黎曼梯度下降求解瑞利商最小化问题
    %
    % 求解问题：
    %   min x^H A x
    %   s.t. ||x||_2 = 1
    %
    % 输入：
    %   A    : Hermitian 矩阵
    %   opts : 参数结构体
    %
    % 输出：
    %   x          : 估计得到的最小特征值对应的特征向量
    %   lambda_est : 估计得到的最小瑞利商
    %   info       : 迭代信息，包括目标函数、梯度范数、步长等

    
    % 保证 A 是 Hermitian 矩阵
    A = (A + A') / 2;
    
    % 初始化到单位球面
    x = x / norm(x);
    
    % 读取参数
    eta0 = opts.eta0;
    rho = opts.rho;
    c = opts.c;
    epsilon = opts.epsilon;
    Kmax = opts.Kmax;
    max_backtrack = opts.max_backtrack;
    eta_min = opts.eta_min;
    
    % 记录迭代历史
    obj_hist = zeros(Kmax, 1);
    grad_hist = zeros(Kmax, 1);
    eta_hist = zeros(Kmax, 1);
    bt_hist = zeros(Kmax, 1);
    
    for k = 1:Kmax
        % 当前瑞利商，由于 ||x|| = 1，所以分母为 1
        f = real(x' * A * x);
    
        % 黎曼梯度，省略常数因子 2
        g = A * x - f * x;
        g_norm = norm(g);
    
        % 记录当前迭代信息
        obj_hist(k) = f;
        grad_hist(k) = g_norm;
    
        % 收敛判断
        if g_norm <= epsilon
            exitflag = 1;
            msg = 'Converged: gradient norm is below tolerance.';
            break;
        end
    
        % Armijo 回溯线搜索
        eta = eta0;
        success = false;
    
        for bt = 1:max_backtrack
            % 沿负黎曼梯度方向更新
            x_new = x - eta * g;
    
            % 回缩到单位球面
            x_new = x_new / norm(x_new);
    
            % 候选点目标函数
            f_new = real(x_new' * A * x_new);
    
            % Armijo 充分下降条件
            if f_new <= f - c * eta * g_norm^2
                success = true;
                break;
            end
    
            % 缩小步长
            eta = rho * eta;
    
            % 数值保护
            if eta <= eta_min
                break;
            end
        end
    
        % 若线搜索失败，通常表示已经接近数值精度极限
        if ~success
            exitflag = 0;
            msg = 'Stopped: Armijo line search failed due to numerical precision.';
            break;
        end
    
        % 接受更新
        x = x_new;
        eta_hist(k) = eta;
        bt_hist(k) = bt;
    end
    
    % 如果达到最大迭代次数
    if k == Kmax
        f = real(x' * A * x);
        g = A * x - f * x;
        g_norm = norm(g);
        obj_hist(k) = f;
        grad_hist(k) = g_norm;
        exitflag = 2;
        msg = 'Stopped: maximum number of iterations reached.';
    end
    
    % 截断历史记录
    obj_hist = obj_hist(1:k);
    grad_hist = grad_hist(1:k);
    eta_hist = eta_hist(1:k);
    bt_hist = bt_hist(1:k);
    
    % 输出最小瑞利商
    lambda_est = real(x' * A * x);
    
    % 保存信息
    info.obj_hist = obj_hist;
    info.grad_hist = grad_hist;
    info.eta_hist = eta_hist;
    info.bt_hist = bt_hist;
    info.iter = k;
    info.exitflag = exitflag;
    info.message = msg;
end