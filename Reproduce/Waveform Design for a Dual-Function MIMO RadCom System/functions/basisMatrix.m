function [Fi_re, Fi_im] = basisMatrix(N, i, j)
    Fi_re = zeros(N, N);
    Fi_re(i, j) = 1;
    Fi_re(j, i) = 1;

    Fi_im = zeros(N, N);
    Fi_im(i, j) = 1i;
    Fi_im(j, i) = -1i;
end