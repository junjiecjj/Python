

% Eq.(57)-(59)
function CRB_rad2 = crb_single_target_equiv_model(theta_deg, alpha, Rs, M, N_eff, sigma2, A_fun, dA_fun)
    d = d_beta_vec(theta_deg, Rs, M, N_eff, A_fun);
    dd = d_beta_deriv_vec(theta_deg, Rs, M, N_eff, dA_fun);
    G = zeros(length(d), 3);
    G(:, 1) = alpha * dd;
    G(:, 2) = d;
    G(:, 3) = 1j * d;
    J = (2 / sigma2) * real(G' * G);
    J = (J + J.') / 2;
    if rcond(J) < 1e-12
        CRB_rad2 = NaN;
    else
        J_inv = J \ eye(3);
        CRB_rad2 = J_inv(1, 1);
    end
end