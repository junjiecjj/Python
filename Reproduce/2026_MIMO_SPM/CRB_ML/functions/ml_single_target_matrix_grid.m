

function theta_hat_deg = ml_single_target_matrix_grid(Y, X, theta_grid_deg, a_fun, v_fun)
    R = X * X' / size(X, 2);
    bestScore = -inf;
    theta_hat_deg = theta_grid_deg(1);
    for idxTheta = 1:length(theta_grid_deg)
        theta_rad = theta_grid_deg(idxTheta) * pi / 180;
        a = a_fun(theta_rad);
        v = v_fun(theta_rad);
        numerator = abs(v' * Y * X' * a)^2;
        denominator = real((v' * v) * (a' * R * a));
        score = numerator / denominator;
        if score > bestScore
            bestScore = score;
            theta_hat_deg = theta_grid_deg(idxTheta);
        end
    end
end