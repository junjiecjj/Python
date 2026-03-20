%% Random transform (RT) or its inverse (IRT)
% x' = Xi * x or x' = Xi^H * x
% where Xi = Pi * U, Pi is a random permutation, U is a fast transform
% ------------------------------------------------------------------
% Input:
% (1) x: vector to be RT/IRT
% (2) type: string
% 'fft': fast Fourier transform (U = ifft)
% 'fwht': fast Walsh–Hadamard transform (U = ifwht)
% (3) is_inv: integer
% 0 means Xi * x, 1 means Xi^H * x
% (4) index: random permutation vector 
% ------------------------------------------------------------------
% Output:
% res: RT/IRT vector
% ------------------------------------------------------------------
function x = Random_transform(s, type, is_inv, index)
    N = length(s);
    if strcmpi(type, 'fft')
        if is_inv
            x = zeros(N, 1);
            x(index) = s;
            x = fft(x) / sqrt(N);
        else
            x = ifft(s) * sqrt(N);
            x = x(index);
        end
    elseif strcmpi(type, 'fwht')
        if is_inv
            x = zeros(N, 1);
            x(index) = s;
            x = fwht(x, N, 'sequency') * sqrt(N);
        else
            x = ifwht(s, N, 'sequency') / sqrt(N);
            x = x(index);
        end
    else
        error('RT Error: "%s" is not supported currently!', type);
    end
end