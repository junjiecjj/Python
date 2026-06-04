
function dd = d_beta_deriv_vec(theta_deg, Rs, M, N_eff, dA_fun)
    U_sqrtL = get_U_sqrtL_rank(Rs, M);
    X = sqrt(N_eff) * dA_fun(theta_deg) * U_sqrtL;
    dd = X(:);
end