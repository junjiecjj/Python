%% Bits to QPSK
% Suppose that length(x_d) is even
function x = Bits_to_QPSK(x_d)
    N = length(x_d);
    x = zeros(N/2, 1);
    k = 1;
    for ii = 1:2:N
        if x_d(ii) == 0 && x_d(ii+1) == 0
            tmp = 1 + 1i;
        elseif x_d(ii) == 0 && x_d(ii+1) == 1
            tmp = 1 - 1i;
        elseif x_d(ii) == 1 && x_d(ii+1) == 1
            tmp = -1 - 1i;
        else
            tmp = -1 + 1i;
        end
        x(k) = tmp;
        k = k + 1;
    end
    x = x / sqrt(2);
end