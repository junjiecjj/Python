%% LE for OAMP (SVD)
function [u_post, v_post] = LE_OAMP(u, v, V, AHy, dia, v_n, M, N)
    rho = v_n / v;
    Dia = [dia.^2; zeros(N-M, 1)];
    D = 1 ./ (rho + Dia);
    u_post = V * (D .* (V' * (AHy + rho * u)));
    v_post = v_n / N * sum(D);
end
