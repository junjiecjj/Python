%By Jianhua Mo

% For K-by-LN matrix X, this gives a fast implementation of X * kron(I, B), 
% for unitary B=kron( dftmtx(sqrt(N)), dftmtx(sqrt(N)) )/sqrt(N) and identity I = eye(L)

% 'Xh' represent the Hermitian of 'X'.

function Y = X_IKronBfast(X,N)

[Nd, LN] = size(X);
L = LN/N;

% idea of the program
%   X * kron(I, B)
% = ( kron(I, B') * X' )'
Xh = X';
XXh = reshape(Xh, [N, L*Nd]);
YYh = Bhfast(XXh,N); 
Yh = reshape(YYh,[LN,Nd]);
Y = Yh';
