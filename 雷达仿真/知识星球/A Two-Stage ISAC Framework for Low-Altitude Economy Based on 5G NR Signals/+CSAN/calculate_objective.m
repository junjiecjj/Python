function obj = calculate_objective(r_bar, S_tilde, z_bar, U, t, M, N, lambda)
    % 辅助函数，用于计算当前的目标函数值 
    obj_l2 = 0.5 * norm(r_bar - S_tilde * z_bar, 2)^2;
    obj_trace_u0 = (lambda / 2) * U(M, N); % u0(0)
    obj_t = (lambda / 2) * t;
    % obj_l1 已移除
    obj = real(obj_l2 + obj_trace_u0 + obj_t);
end
