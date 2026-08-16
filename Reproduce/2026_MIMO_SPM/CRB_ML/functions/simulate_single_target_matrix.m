

function Y = simulate_single_target_matrix(theta_rad, beta, X, sigma2, a_fun, v_fun)
    M = length(v_fun(theta_rad));
    L = size(X, 2);
    a = a_fun(theta_rad);
    v = v_fun(theta_rad);
    Z = sqrt(sigma2 / 2) * (randn(M, L) + 1j * randn(M, L));
    Y = beta * v * a' * X + Z;
end
