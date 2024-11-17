% When X is a vector or matrix, this gives a fast implementation of B*X, 
% for unitary B=kron(dftmtx(N),dftmtx(N))/N;
%
% When X is a multidimensional array, this performs the above multiplication
% on the first dimension.

function BX = Bfast(X,N)

sqrtN = sqrt(N);
sizeX = size(X);
sizeX2 = sizeX(2:end); 
BX = reshape( (1/sqrtN)*fft2(reshape(X,[sqrtN,sqrtN,sizeX2])) ,[N,sizeX2]);

