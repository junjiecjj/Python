%% MLE for BO-MAMP (do not know the singular values of A)
% Suppose that we do not know "dia". ("dia" is only used in Ax and AHx)
function [log_theta_, r_hat, r, v_gamma] = MLE_MAMP_e(x_phi, v_phi, r_hat, log_theta_, sign_w, ...
    log_w, w_0, w_bar_00, z, A, lambda_s, t, v_n, M, N)
    p_bar = zeros(1, t);
    theta = 1 / (lambda_s + v_n / v_phi(t, t));
    log_theta_(1:t-1) = log_theta_(1:t-1) + log(theta);
    p_bar(1:t-1) = fliplr(sign_w(2:t)) .* exp(log_theta_(1:t-1) + fliplr(log_w(2:t)));
    [c0, c1, c2, c3] = Get_c(p_bar, v_phi, sign_w, log_w, log_theta_, w_0, w_bar_00, lambda_s, v_n, t);
    if t > 1
        xi = (c2 * c0 + c3) / (c1 * c0 + c2);
    else
        xi = 1;
    end
    log_theta_(t) = log(xi);
    p_bar(t) = xi * w_0;
    epsilon = (xi + c0) * w_0;
    v_gamma = (c1 * xi^2 - 2 * c2 * xi + c3) / epsilon^2;
    % r_hat and r
    AHr_ = AH_times_x(r_hat, A);
    AAHr_ = A_times_x(AHr_, A);
    r_hat = xi * z(:, t) + theta * (lambda_s * r_hat - AAHr_);
    AHr_ = AH_times_x(r_hat, A);
    tmp = 0;
    for i = 1 : t
        tmp = tmp + p_bar(i) * x_phi(:, i);
    end
    r = 1 / epsilon * (AHr_ + tmp);
end

%% c0, c1, c2, c3
function [c0, c1, c2, c3] = Get_c(p_bar, v_phi, sign_w, log_w, log_theta, w_0, w_bar_00, lambda_s, v_n, t)
    c0 = sum(p_bar(1:t-1)) / w_0;
    c1 = v_n * w_0 + v_phi(t, t) * w_bar_00;
    c2 = 0;
    for i = 1 : t-1
        v_ti = real(v_phi(t, i));
        tmp = sign_w(t-i+1) * exp(log_theta(i)+log_w(t-i+1));
        c2 = c2 - (v_n + lambda_s * v_ti) * tmp;
        c2 = c2 + v_ti * sign_w(t-i+2) * exp(log_theta(i)+log_w(t-i+2));
        c2 = c2 + v_ti * w_0 * tmp;
    end
    c3 = 0;
    for i = 1 : t-1
        for k = 1 : t-1
            v_ij = real(v_phi(i, k));
            c3 = c3 + (v_n + lambda_s*v_ij) * sign_w(2*t-i-k+1) * ...
                exp(log_theta(i)+log_theta(k)+log_w(2*t-i-k+1));
            c3 = c3 - v_ij * sign_w(2*t-i-k+2) * exp(log_theta(i)+log_theta(k)+log_w(2*t-i-k+2));
            c3 = c3 - v_ij * sign_w(t-i+1) * sign_w(t-k+1) * ...
                exp(log_theta(i)+log_theta(k)+log_w(t-i+1)+log_w(t-k+1));
        end
    end
end

%% Ax
function Ax = A_times_x(x, A)
    A1=sparse(A);
    Ax = A1*x;
end

%% AHx
function AHx = AH_times_x(x, A)
    A2=sparse(A');
    AHx = A2*x;
end