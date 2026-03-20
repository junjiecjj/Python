%% MLE for MAMP
% log_th1 = log(theta_1)
function [log_vth, sgn_xi, u, r_hat, r, v_gam] = MLE_MAMP(A, x_phi, v_phi, chi, sgn_xi, log_vth, ...
            y, u, r_hat, w_0, wb_00, lam_s, v_n, log_th1, t)
    p = zeros(1, t);
    theta = 1 / (lam_s + v_n / v_phi(t, t));
    if t > 1
        log_vth(1:t-1) = log_vth(1:t-1) + log(theta);
        p(1:t-1) = chi(flip(1:t-1)) .* sgn_xi(1:t-1) .* exp(log_vth(1:t-1)-log_th1*flip(1:t-1));
        % c1, c2, c3, c4
        [c0, c1, c2, c3] = Get_c(v_phi, p, chi, sgn_xi, log_vth, v_n, w_0, wb_00, lam_s, log_th1, t);
        tmp = c1 * c0 + c2;
        if tmp ~= 0
            xi = (c2 * c0 + c3) / tmp;
        else
            xi = 1 / (v_phi(t, t) + v_n);
        end
    else
        [c2, c3] = deal(0);
        c1 = v_n * w_0 + v_phi(1, 1) * wb_00;
        xi = 1 / (v_phi(t, t) + v_n);
    end
    % xi = 1 / (v_phi(t, t) + v_n);     % simple but close to be optimal 
    sgn_xi(t) = sign(xi);
    log_vth(t) = log(abs(xi));
    p(t) = xi * w_0;
    eps = sum(p(1:t));
    u = theta * lam_s * u + xi * y - A * (theta * r_hat + xi * x_phi(:, t));
    r_hat = A' * u;
    temp = 0;
    for i = 1 : t
        temp = temp + p(i) * x_phi(:, i);
    end
    r = 1 / eps * (r_hat + temp);
    v_gam = real((c1 * xi^2 - 2 * c2 * xi + c3) / eps^2);
    % N = length(x_phi(:, 1));
    % v_gam = abs((r - x_phi(:, t))' * (r - x_phi(:, t)) / N - v_phi(t, t)); 
end


%% c0, c1, c2, c3
function [c0, c1, c2, c3] = Get_c(v_phi, p, chi, sgn_xi, log_vth, v_n, w_0, wb_00, lam_s, log_th1, t)
    v_phi = real(v_phi);
    % c0
    c0 = sum(p(1:t-1)) / w_0;
    % c1
    c1 = v_n * w_0 + v_phi(t, t) * wb_00;
    % c2
    coef_1 = (w_0 - lam_s) * v_phi(t, 1:t-1) - v_n;
    term_2 = chi(flip(2:t)).*sgn_xi(1:t-1).*exp(log_vth(1:t-1)-log_th1*flip(2:t));
    c2 = sum(coef_1 .* p(1:t-1) + v_phi(t, 1:t-1) .* term_2);
    % c3
    c3 = 0;
    for ii = 1 : t-1
        for jj = 1 : t-1
            g = 2 * t - ii - jj;
            s = sgn_xi(ii) * sgn_xi(jj);
            l = log_vth(ii) + log_vth(jj);
            coef = (v_n + v_phi(ii, jj) * lam_s) * chi(g);
            coef = coef - v_phi(ii, jj) * (chi(g+1)/exp(log_th1) + chi(t-ii)*chi(t-jj));
            c3 = c3 + coef * s * exp(l-g*log_th1);
        end
    end
end
