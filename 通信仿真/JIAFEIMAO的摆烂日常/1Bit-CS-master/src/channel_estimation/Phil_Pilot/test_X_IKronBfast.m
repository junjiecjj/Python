% tests fast implementation of X * kron(I, B) where B=kron(DFT,DFT) and X is a matrix

% Author: Jianhua Mo
% Feb, 24th, 2016

N = 16;
L = 4;

% explicit representation of 2D-ULA matrix
sqN = sqrt(N);
F = (1/sqrt(sqN))*dftmtx(sqN);
B = kron(F,F);
I = eye(L);
A = kron(I, B);

% test fast multiplication by vector
x = randn(1, N*L)+1i*randn(1, N*L);
xA = x * A;
%Ax_fast = reshape( reshape(x,L,N)*B' ,N*L,1);
%Ax_fast = reshape( ( Bfast(reshape(x,L,N)',N) )' ,N*L,1);
xA_fast = IKronBfast(x,N);
max(abs(xA - xA_fast))

% test fast multiplication by matrix
Nd = 2000;
X = randn(Nd , N*L)+1i*randn(Nd, N*L);

for ii=1:1:1e2
    XA = X * A;
end;

for ii=1:1:1e2
    XA_fast = IKronBfast(X,N);
end;
max(abs(XA(:)-XA_fast(:)))
