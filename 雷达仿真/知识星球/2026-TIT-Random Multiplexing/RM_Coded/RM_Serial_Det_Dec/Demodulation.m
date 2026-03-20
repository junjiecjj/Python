%% QPSK
function [u_post, v_post] = Demodulation(u, v, N) 
    u_re = sqrt(2) * real(u);              
    u_im = sqrt(2) * imag(u);
    [u_1, v_1] = Demod(u_re, v, 0.5, N);    % real part
    [u_2, v_2] = Demod(u_im, v, 0.5, N);    % imaginary part
    u_post = (u_1 + u_2*1i) / sqrt(2);
    v_post = (v_1 + v_2) / 2;
end

%% y = x + n, x /in {1, -1} with p_x(1) = p_1
function [u_post, v_post] = Demod(u, v, p1, N)
    thres = 1e-9;
    if v < thres
        u_post = u;
        v_post = 0;
        return
    end
    u_post = zeros(N, 1);
    v_post = 0;
    for i = 1 : N
        p_1 = p1 / (p1 + (1 - p1) * exp(-2 * u(i) / v));
        u_post(i) = 2*p_1 - 1;
        v_post = v_post + (1 - u_post(i)^2);
    end
    v_post = v_post / N;
    if v_post < thres
        v_post = thres;
    end
end

