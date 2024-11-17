% For LN-by-K matrix X, this gives a fast implementation of kron(B',I)*X, 
% for unitary B=kron(dftmtx(N),dftmtx(N))/N and identity I = eye(L)

function Y = BhKronIfast(X,N)

[LN,K] = size(X);
L = LN/N;

XX = reshape(X,L,N,K); % form tensor
XXh = conj(permute(XX,[2,1,3])); % conj transpose of first two dimensions
YYh = Bfast(XXh,N); % apply B 
YY = conj(permute(YYh,[2,1,3])); % conj transpose of first two dimensions
Y = reshape(YY,[LN,K]); % form matrix
