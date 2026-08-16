



function R_opt = MinimumSidelobeDesignWithNull(c, M, theta0, theta1, theta2, Omega, theta_null, null_level_dB)
    fprintf('求解 Fig.4(c) minimum sidelobe design with null...\n');

    N_Omega = length(Omega);
    a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));

    a0 = a(theta0);
    a1 = a(theta1);
    a2 = a(theta2);
    an = a(theta_null);

    a_Omega = zeros(M, N_Omega);
    for k = 1:N_Omega
        a_Omega(:, k) = a(Omega(k));
    end

    null_ratio = 10^(null_level_dB / 10);

    cvx_begin sdp quiet
        variable R(M, M) hermitian
        variable t
        expression P0

        P0 = real(a0' * R * a0);

        maximize t
        subject to
            for k = 1:N_Omega
                real(a0' * R * a0 - a_Omega(:, k)' * R * a_Omega(:, k)) >= t;
            end

            real(a1' * R * a1) == 0.5 * P0;
            real(a2' * R * a2) == 0.5 * P0;
            real(an' * R * an) <= null_ratio * P0;
            R == hermitian_semidefinite(M);
            sum(real(diag(R))) == sum(c);

            for m = 1:M
                real(R(m, m)) >= 0.8 * c(m);
                real(R(m, m)) <= 1.2 * c(m);
            end
    cvx_end

    if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
        fprintf('求解成功，最优 t = %f\n', t);
        R_opt = R;
    else
        error('CVX 求解失败: %s', cvx_status);
    end
end