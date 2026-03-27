function x = p4(N)
% x = p4(N), generate a P4 sequence of length N

n = (1:N)';
phi = 2*pi/N * (n-1) .* (n-1-N)/2;
x = exp(1i * phi);