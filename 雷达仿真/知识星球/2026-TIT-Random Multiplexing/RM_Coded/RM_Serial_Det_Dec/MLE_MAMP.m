%% MLE for MAMP
% ------------------------------------------------------------------------
function [log_theta_, theta_w_, r_hat, r, v_gamma] = MLE_MAMP(H, x_phi, v_phi, log_theta_, theta_w_, ...
            z, r_hat, B, sign, log_B, w_0, w_bar_00, lambda_s, t, v_n, N)
    p_bar = zeros(1, t);
    theta = 1 / (lambda_s + v_n / v_phi(t, t));
    if t > 1
        log_theta_p = log_theta_(1:t-1);
        log_theta_(1:t-1) = log_theta_p(1:t-1) + log(theta);
        theta_w_(1:t-2) = theta_w_(1:t-2) .* exp(fliplr(log_theta_(2:t-1) - log_theta_p(1:t-2)));
        theta_w_(t-1) = theta_w(lambda_s, B, sign, log_B, log_theta_(1), t-1, N);
        theta_w_(t) = theta_w(lambda_s, B, sign, log_B, log_theta_(1), t, N);
        log_theta_11 = log_theta_(1) + log_theta_(1);
        theta_w_(2*t-1) = theta_w(lambda_s, B, sign, log_B, log_theta_11, 2*t-1, N);
        if t > 2
            log_theta_12 = log_theta_(1) + log_theta_(2);
            theta_w_(2*t-2) = theta_w(lambda_s, B, sign, log_B, log_theta_12, 2*t-2, N);
            if t > 3
                theta_w_(t+1:2*t-3) = theta_w_(t+1:2*t-3) .* ...
                    exp(log(theta) + fliplr(log_theta_(3:t-1) - log_theta_p(1:t-3)));
            end
        end
        p_bar(1:t-1) = fliplr(theta_w_(1:t-1));
        [c0, c1, c2, c3] = Get_c(p_bar, v_phi, log_theta_, theta_w_, v_n, w_0, w_bar_00, lambda_s, t);
        tmp = c1 * c0 + c2;
        if tmp ~= 0
            xi = (c2 * c0 + c3) / tmp;
        else
            xi = 1;
        end
    else
        [c0, c2, c3] = deal(0);
        c1 = v_n * w_0 + v_phi(1, 1) * w_bar_00;
        xi = 1;
    end
    log_theta_(t) = log(xi);
    p_bar(t) = xi * w_0;
    epsilon = (xi + c0) * w_0;
    v_gamma = (c1 * xi^2 - 2 * c2 * xi + c3) / epsilon^2;
    % r_hat and r
    r_hat = xi * z(:, t) + theta * (lambda_s * r_hat - H*(H'*r_hat));
    temp = 0;
    for i = 1 : t
        temp = temp + p_bar(i) * x_phi(:, i);
    end
    r = 1 / epsilon * (H'*r_hat + temp);
end



