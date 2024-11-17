% tests fast implementation of kron(B',I)*X where B=kron(DFT,DFT) and X is a matrix
N = 16;
L = 4;

% explicit representation of 2D-ULA matrix
sqN = sqrt(N);
F = (1/sqrt(sqN))*dftmtx(sqN);
B = kron(F,F);
I = eye(L);
A = kron(B',I);

% test fast multiplication by vector
x = randn(N*L,1)+1i*randn(N*L,1);
Ax = A*x;
%Ax_fast = reshape( reshape(x,L,N)*B' ,N*L,1);
%Ax_fast = reshape( ( Bfast(reshape(x,L,N)',N) )' ,N*L,1);
Ax_fast = BhKronIfast(x,N);
max(abs(A*x-Ax_fast))

% test fast multiplication by matrix
K = 2;
X = randn(N*L,K)+1i*randn(N*L,K);
AX = A*X;
AX_fast = BhKronIfast(X,N);
max(abs(AX(:)-AX_fast(:)))
