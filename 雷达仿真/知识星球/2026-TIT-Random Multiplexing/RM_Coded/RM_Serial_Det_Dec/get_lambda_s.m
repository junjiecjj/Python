%% lambda_s
function lambda_s = get_lambda_s(A, M, N, it, L1)
    R = 2 * it;
    thres = 1e250;
    lam = 0;
    tau = 0;
    for k = 1 : L1
        if k == 2 && tau < R
            R = tau;
        end
        s0_re = normrnd(0, 1, [N, 1]);
        s0_im = normrnd(0, 1, [N, 1]);
        s0 = s0_re + s0_im*1i;
        s0 = s0 / norm(s0);
        s_i = s0;
        for i = 1 : R
            if mod(i,2) == 1
                s_i = A_times_x(s_i, A, M, N);
            else
                s_i = AH_times_x(s_i, A, M, N);
            end
            tmp = real(s_i' * s_i);
            if i == R || (k == 1 && tmp > thres)
                tau = i;
                lam = lam + tmp;
                break
            end
        end
    end
    lam = lam / L1;                             
    lambda_s = 0.5 * (N * lam)^(1/R);
end

%% Ax
function Ax = A_times_x(x, A, M, N)
    Ax = A*x;
end

%% AHx
function AHx = AH_times_x(x, A, M, N)
    AHx = A'*x;
end
