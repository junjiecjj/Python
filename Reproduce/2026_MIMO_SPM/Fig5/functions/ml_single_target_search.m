

function theta_hat = ml_single_target_search(z, beta, M, N_eff, A_fun, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width)
    theta_grid = theta_search_min:coarse_step:theta_search_max;
    theta_hat = ml_single_target_grid(z, beta, M, N_eff, A_fun, theta_grid);
    theta_grid = theta_hat - fine_width:fine_step:theta_hat + fine_width;
    theta_grid = theta_grid(theta_grid >= theta_search_min);
    theta_grid = theta_grid(theta_grid <= theta_search_max);
    theta_hat = ml_single_target_grid(z, beta, M, N_eff, A_fun, theta_grid);
end
