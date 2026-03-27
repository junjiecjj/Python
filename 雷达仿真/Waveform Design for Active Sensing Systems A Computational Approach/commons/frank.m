function x = frank(M)
% x = frank(M), generate a Frank sequence x, length N=M^2

N = M^2;
x = zeros(N, 1);

for n = 0:(M-1)
    for k = 0:(M-1)
        phi = 2*pi * n * k / M;
        x(n*M+k+1) = exp(1i * phi);
    end
end