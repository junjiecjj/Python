
function CRB_rad2 = crb_single_target_eqA(theta_rad, beta, R, L, sigma2, a_fun, v_fun, da_fun, dv_fun)
    a = a_fun(theta_rad);
    v = v_fun(theta_rad);
    adot = da_fun(theta_rad);
    vdot = dv_fun(theta_rad);
    H = v * a';
    Hdot = vdot * a' + v * adot';
    % F11 = real(trace(Hdot * R * Hdot'));
    F11 =  trace(Hdot * R * Hdot');
    F12 = trace(H * R * Hdot');
    F22 = real(trace(H * R * H'));
    denominator = 2 * L * abs(beta)^2 * (F11 * F22 - abs(F12)^2);
    CRB_rad2 = sigma2 * F22 / denominator;
end
