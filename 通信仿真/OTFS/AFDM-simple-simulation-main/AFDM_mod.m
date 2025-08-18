% Description: AFDM Modulation
% x: input data vector
% c1, c2: AFDM parameters

function [out] = AFDM_mod(x, c1, c2)

N = size(x,1);
F = dftmtx(N);
F = F./norm(F);
L1 = diag(exp(-1i*2*pi*c1*((0:N-1).^2)));
L2 = diag(exp(-1i*2*pi*c2*((0:N-1).^2)));
A = L2*F*L1;
out = A'*x;

end