%% log(|w|)
function [log_w, sign_w] = get_logw(A, lambda_s, M, N, it, L2, v_n)                       
    log_w = zeros(1, 2*it);
    sign_w = ones(1, 2*it);
    sp = zeros(1, 2*it);
    for k = 1 : L2
        s0_re = normrnd(0, 1, [N, 1]);
        s0_im = normrnd(0, 1, [N, 1]);
        s0 = s0_re + s0_im*1i;
        s0 = s0 / norm(s0);
        s0 = A_times_x(s0, A, M, N);
        s_t = s0;
        for t = 1 : 2*it
            sp(t) = sp(t) + real(s0'*s_t);
            if t == 2*it
                break
            end
            tmp = AH_times_x(s_t, A, M, N);
            tmp = A_times_x(tmp, A, M, N);
            s_t = lambda_s * s_t / (lambda_s + v_n) - tmp / (lambda_s + v_n);
        end
    end
    sp = sp / L2;
    for t = 1 : 2*it
        if sp(t) < 0
            sign_w(t) = -1;
        end
        log_w(t) = log(abs(sp(t))) + (t-1)*log(lambda_s+v_n);
    end
end

%% Ax
function Ax = A_times_x(x, A, M, N)
    Ax = A*x;
end

%% AHx
function AHx = AH_times_x(x, A, M, N)
    AHx = A'*x;
end
