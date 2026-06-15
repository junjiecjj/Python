

function theta_hat_deg = ml_single_target_matrix_search(Y, X, theta_min_deg, theta_max_deg, coarse_step_deg, fine_step_deg, fine_width_deg, a_fun, v_fun)
    theta_grid_deg = theta_min_deg:coarse_step_deg:theta_max_deg;
    theta_hat_deg = ml_single_target_matrix_grid(Y, X, theta_grid_deg, a_fun, v_fun);
    theta_grid_deg = theta_hat_deg - fine_width_deg:fine_step_deg:theta_hat_deg + fine_width_deg;
    theta_grid_deg = theta_grid_deg(theta_grid_deg >= theta_min_deg);
    theta_grid_deg = theta_grid_deg(theta_grid_deg <= theta_max_deg);
    theta_hat_deg = ml_single_target_matrix_grid(Y, X, theta_grid_deg, a_fun, v_fun);
end