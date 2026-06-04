
% 
function theta_hat = ml_single_target_grid(z, beta, M, N_eff, A_fun, theta_grid)
    best_score = -inf;
    theta_hat = theta_grid(1);
    for k = 1:length(theta_grid)
        d = d_beta_vec(theta_grid(k), beta, M, N_eff, A_fun);
        score = abs(d' * z)^2 / real(d' * d);  % Eq.(29)
        if score > best_score
            best_score = score;
            theta_hat = theta_grid(k);
        end
    end
end