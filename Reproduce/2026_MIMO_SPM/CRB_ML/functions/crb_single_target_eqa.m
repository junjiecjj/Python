
function CRB_rad2 = crb_single_target_eqa(theta_rad, beta, R, L, sigma2, a_fun, v_fun, da_fun, dv_fun)
    a = a_fun(theta_rad);
    v = v_fun(theta_rad);
    adot = da_fun(theta_rad);
    vdot = dv_fun(theta_rad);
    M = length(v);
    aRa = (a' * R * a);
    adotRadot = (adot' * R * adot);
    aRadot = a' * R * adot;
    vdotNorm2 = (vdot' * vdot);
    denominatorInner = aRa * vdotNorm2 + M * (adotRadot - abs(aRadot)^2 / aRa);
    CRB_rad2 = sigma2 / (2 * L * abs(beta)^2 * denominatorInner);
end
