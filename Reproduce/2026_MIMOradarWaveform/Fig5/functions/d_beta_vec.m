function d = d_beta_vec(theta_deg, Rs, M, N_eff, A_fun)
    U_sqrtL = get_U_sqrtL_rank(Rs, M);
    X = sqrt(N_eff) * A_fun(theta_deg) * U_sqrtL;
    d = X(:);
end