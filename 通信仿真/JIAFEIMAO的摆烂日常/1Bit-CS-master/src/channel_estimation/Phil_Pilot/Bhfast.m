% When X is a vector or matrix, this gives a fast implementation of B'*X, 
% for unitary B=kron(dftmtx(N),dftmtx(N))/N;
%
% When X is a multidimensional array, this performs the above multiplication
% on the first dimension.

% Author: Jianhua Mo
% Feb, 24th, 2016

function BhX = Bhfast(X,N)

sqrtN = sqrt(N);
sizeX = size(X);
sizeX2 = sizeX(2:end); 
BhX = reshape( sqrtN * ifft2(reshape(X,[sqrtN,sqrtN,sizeX2])) ,[N,sizeX2]);



