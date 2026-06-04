

% Eq.(15), compute_eta
function z = simulate_single_target_equiv(theta_deg, alpha, beta, M, N_eff, sigma2, A_fun)
    d = d_beta_vec(theta_deg, beta, M, N_eff, A_fun);
    w = sqrt(sigma2 / 2) * (randn(size(d)) + 1j * randn(size(d)));
    z = alpha * d + w;
end
