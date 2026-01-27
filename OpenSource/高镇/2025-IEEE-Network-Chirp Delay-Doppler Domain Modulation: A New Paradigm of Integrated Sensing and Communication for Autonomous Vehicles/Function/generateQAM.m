function [symbols, bits] = generateQAM(N)
% input
%   N           order of QAM
% output
%   symbols     Nx1 QAM symbols
%   bits        Nxlog2(N), each row corresponding to bits seq. of the
%               associated symbol
    k = sqrt(N);
    b_axis = log2(k);           % number of bits in each axis
    coords = -(k-1):2:(k-1);
    [X, Y] = meshgrid(coords, coords);
    constellation = X(:) + 1j*Y(:);
    P_avg = mean(abs(constellation).^2);
    symbols = constellation / sqrt(P_avg);

     % generate Gray code
    levels = 0:(k-1);                              % 0,1,...,k-1
    gray_codes = bitxor(levels, floor(levels/2)); % binary Gray code
    % Convert to the binary matrix with b_axis columns, with MSB on the left
    gray_mat = de2bi(gray_codes, b_axis, 'left-msb');

    % map I/Q coordinate to Gray bits
    % coordinate->index：level = (coord + (k-1)) / 2   -> range 0:(k-1)
    idx_I = round(( real(constellation) + (k-1) )/2) + 1; 
    idx_Q = round(( imag(constellation) + (k-1) )/2) + 1; 
    % combination：I-axis bit first，then Q-axis
    bits = [ gray_mat(idx_I, :) , gray_mat(idx_Q, :) ];
end