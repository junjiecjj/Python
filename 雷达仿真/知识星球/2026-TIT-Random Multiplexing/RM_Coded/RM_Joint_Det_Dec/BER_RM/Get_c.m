%% Calculate c0, c1, c2, and c3
function [c0, c1, c2, c3] = Get_c(p_bar, v_phi, log_theta_, theta_w_, v_n, w_0, w_bar_00, lambda_s, t)
    % c0
    c0 = sum(p_bar(1:t-1)) / w_0;
    % c1
    c1 = v_n * w_0 + v_phi(t, t) * w_bar_00;
    % c2
    term_1 = p_bar(1:t-1);
    temp = real(v_phi(t, 1:t-1));
    coeff_1 = v_n + temp * (lambda_s - w_0);
    term_2 = zeros(1, t-1);
    term_2(1) = theta_w_(t);
    term_2(2:t-1) = p_bar(1:t-2) .* exp(log_theta_(2:t-1) - log_theta_(1:t-2));
    c2 = sum(temp .* term_2 - coeff_1 .* term_1);
    % c3
    c3 = 0;
    for i = 1 : t-1
        for j = 1 : t-1
            if 2*t-i-j < t 
                coffe_1 = exp(log_theta_(i) + log_theta_(j) - log_theta_(i+j-t));
            elseif 2*t-i-j == t
                coffe_1 = exp(log_theta_(i) + log_theta_(j) - log_theta_(1));
            else
                coffe_1 = exp(log_theta_(i) + log_theta_(j) - log_theta_(1) - log_theta_(i+j));
            end
            term_1 = (v_n + v_phi(i, j) * lambda_s) * coffe_1 * theta_w_(2*t-i-j);
            if 2*t-i-j+1 < t 
                coffe_2 = exp(log_theta_(i) + log_theta_(j) - log_theta_(i+j-t-1));
            elseif 2*t-i-j+1 == t
                coffe_2 = exp(log_theta_(i) + log_theta_(j) - log_theta_(1));
            else
                coffe_2 = exp(log_theta_(i) + log_theta_(j) - log_theta_(1) - log_theta_(i+j-1));
            end
            term_2 = v_phi(i, j) * coffe_2 * theta_w_(2*t-i-j+1);
            term_3 = v_phi(i, j) * p_bar(i) * p_bar(j);
            c3 = c3 + term_1 - term_2 - term_3;
        end
    end
    c3 = real(c3);
end