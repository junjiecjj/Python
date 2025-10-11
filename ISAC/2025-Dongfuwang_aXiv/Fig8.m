clear; clc; close all;

% 参数设置
Nt = 2;
Mc = 2;
Ms = 2;
Iter = 100;               % 信道遍历次数
max_iterations = 100;     % SCA 迭代次数
I_mc = eye(Mc);
I_nt = eye(Nt);
Pt = 1;
K = 100;
SNRcom = [-5, 0, 5];
SNRsen = 0:2:20;
Sigma_s = diag([0.4, 0.1]);

% 预先计算常数矩阵
Sigma_s_inv = inv(Sigma_s);

% 生成随机信道矩阵 H_c
 
H_c = randn(Mc, Nt);

% 设置感知和通信噪声参数
sigma_s = 0.5;
sigma_c = 0.5;

fprintf('System Parameters:\n');
fprintf('Nt = %d, Mc = %d, Ms = %d\n', Nt, Mc, Ms);
fprintf('sigma_s = %.2f, sigma_c = %.2f, Pt = %.2f\n', sigma_s, sigma_c, Pt);

% 初始化 R_0
R_0 = (Pt / Nt) * I_nt;

% 调用solve_P2函数
[R_x_opt, obj_values] = solve_P2(R_0, H_c, sigma_s, sigma_c, Sigma_s_inv, max_iterations, 1e-4);

% 评估解
[D_s, C] = evaluate_solution(R_x_opt, H_c, sigma_s, sigma_c, Sigma_s, Sigma_s_inv, Pt, Ms, Nt, I_mc);

% 绘制收敛曲线
if length(obj_values) > 1
    figure;
    plot(obj_values, 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
    xlabel('Iteration');
    ylabel('Objective Value');
    title('SCA Convergence');
    grid on;
end

% 函数定义
function [R_x_opt, obj_values] = solve_P2(initial_R_0, H_c, sigma_s, sigma_c, Sigma_s_inv, max_iterations, tolerance)
    R_x_prev = initial_R_0;
    obj_values = [];

    for iteration = 1:max_iterations
        [R_x_opt, obj_value] = solve_P3(R_x_prev, H_c, sigma_s, sigma_c, Sigma_s_inv);

        if isempty(R_x_opt)
            fprintf('Iteration %d: No solution found.\n', iteration);
            break;
        end

        obj_values = [obj_values, obj_value];

        fprintf('Iteration %d: Objective = %.6f\n', iteration, obj_value);

        if iteration > 1
            relative_change = abs(obj_value - obj_values(iteration-1)) / (abs(obj_values(iteration-1)) + 1e-8);
            if relative_change < tolerance
                fprintf('Converged after %d iterations.\n', iteration);
                break;
            end
        end

        R_x_prev = R_x_opt;
    end
end

function [R_x_opt, obj_value] = solve_P3(R_0, H_c, sigma_s, sigma_c, Sigma_s_inv)
    [Nt, ~] = size(R_0);
    Ms = 2;  % 注意：这里需要获取全局变量Sigma_s，但MATLAB函数中不能直接使用，所以通过参数传递或定义为全局变量。这里我们通过参数传递。
    % 由于在函数内部无法直接使用外部参数，我们将Sigma_s和Sigma_s_inv作为全局变量，或者通过嵌套函数共享。这里我们选择通过参数传递，但注意调整函数接口。
    % 但是，上面的solve_P2并没有传递Sigma_s，所以我们需要调整。
    % 为了简化，我们将在主程序中定义这些参数，然后在函数中通过共享 workspace 或者使用嵌套函数。这里我们使用嵌套函数的方式，所以将整个代码写在一个文件中。
    % 因此，我们使用主程序中定义的 Sigma_s 和 Sigma_s_inv。
    % 但是，在MATLAB中，嵌套函数可以共享外部函数的变量。所以，我们可以这样写：

    % 计算常数矩阵 P
    inner_matrix_P = (1/sigma_s^2) * R_0 + Sigma_s_inv;
    P = inv(inner_matrix_P);

    cvx_begin sdp quiet
        variable R_x(Nt, Nt) symmetric
        variable D(Ms, Ms) symmetric

        % 计算线性化近似 ~R_s
        constant_part = Sigma_s - P - (1/sigma_s^2) * P * R_0 * P;
        linear_part = (1/sigma_s^2) * P * R_x * P;
        R_s_tilde = constant_part + linear_part;

        % 计算 f(R_x) 的线性化近似
        Sigma_minus_P = Sigma_s - P;
        Sigma_minus_P_inv = inv(Sigma_minus_P);
        f_constant = log(det(Sigma_minus_P)) - (1/sigma_s^2) * trace(Sigma_minus_P_inv * P * R_0 * P);
        f_linear = (1/sigma_s^2) * trace(Sigma_minus_P_inv * P * R_x * P);
        f_Rx = f_constant + f_linear;

        % 目标函数
        inner_matrix_obj = (1/sigma_s^2) * R_x + Sigma_s_inv;
        objective = Ms * trace(inv(inner_matrix_obj)) + trace(D);

        % 约束条件
        inner_matrix_capacity = (1/sigma_c^2) * H_c * R_x * H_c' + I_mc;
        capacity_constraint = log_det(inner_matrix_capacity) - Ms * (f_Rx - log_det(D)) >= 0;

        constraints = [
            capacity_constraint,
            R_s_tilde - D >= 0,
            R_x >= 0,
            trace(R_x) <= Pt
        ];

        minimize(objective)
        subject to
            constraints
    cvx_end

    if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
        R_x_opt = R_x;
        % 计算目标函数值（使用实际的感知失真，而不是CVX中的近似，因为CVX中我们用了inv，但实际目标函数是相同的表达式）
        obj_value = Ms * trace(inv((1/sigma_s^2) * R_x_opt + Sigma_s_inv)) + trace(D);
    else
        R_x_opt = [];
        obj_value = inf;
    end
end

function [D_s, C, R_s] = evaluate_solution(R_x_opt, H_c, sigma_s, sigma_c, Sigma_s, Sigma_s_inv, Pt, Ms, Nt, I_mc)
    inner_matrix = (1/sigma_s^2) * R_x_opt + Sigma_s_inv;
    D_s = Ms * trace(inv(inner_matrix));

    inner_matrix_capacity = (1/sigma_c^2) * H_c * R_x_opt * H_c' + I_mc;
    C = log2(det(inner_matrix_capacity));

    R_s = Sigma_s - inv(inner_matrix);

    fprintf('\n=== Solution Evaluation ===\n');
    fprintf('Sensing Distortion D_s: %.6f\n', D_s);
    fprintf('Channel Capacity C: %.6f\n', C);
    fprintf('Power Constraint: %.6f <= %.6f\n', trace(R_x_opt), Pt);
    fprintf('R_x positive definite: %s\n', mat2str(all(eig(R_x_opt) > 0)));
end
