%% Hard decision + QPSK to bits 
% x_s: hard decision result (QPSK)
% d_s: hard decision result (bits)
function [d_s, x_s] = QPSK_to_bits(x_est)
    T = length(x_est);
    x_s = zeros(T, 1);
    d_s = zeros(2*T, 1);
    k = 1;
    for ii = 1 : T
        if real(x_est(ii)) >= 0 && imag(x_est(ii)) >= 0
            x_s(ii) = 1 + 1i; 
            d_s(k) = 0; 
            d_s(k+1) = 0;
        elseif real(x_est(ii)) >= 0 && imag(x_est(ii)) < 0
            x_s(ii) = 1 - 1i; 
            d_s(k) = 0; 
            d_s(k+1) = 1;
        elseif real(x_est(ii)) < 0 && imag(x_est(ii)) < 0
            x_s(ii) = -1 - 1i; 
            d_s(k) = 1; 
            d_s(k+1) = 1;    
        else
            x_s(ii) = -1 + 1i; 
            d_s(k) = 1; 
            d_s(k+1) = 0;    
        end
        k = k + 2;
    end
    x_s = x_s / sqrt(2);
end