% Tests fast implementation of B' * X where B=kron(DFT,DFT) and X is a matrix
% JMo: When x is a vector, the results show that the fast operations actully are slower than the
% direct mutliplication. :-(. However, when X is a matrix with multiple
% columns, the fast implement saves a lot of time. :-)

% Author: Jianhua Mo
% Feb, 24th, 2016

Nr = 64;

% explicit representation of 2D-ULA matrix
sqNr = sqrt(Nr);
F = (1/sqrt(sqNr))*dftmtx(sqNr);
B = kron(F,F);

%% test fast multiplication by vector
x = randn(Nr,1)+1i*randn(Nr,1);
tic;
Bhx = B' * x;
toc;

tic;
%Bx_fast = reshape( (1/sqNr)*fft2(reshape(x,sqNr,sqNr)) ,Nr,1 );
Bhx_fast = Bhfast(x,Nr);
toc;

max(abs(Bhx-Bhx_fast))

%% test fast multiplication by matrix
L = 2000;
X = randn(Nr,L)+1i*randn(Nr,L);

tic;
for i=1:1:1e1
    BhX = B'*X;
end;
toc;

tic;
for i=1:1:1e1
    %BX_fast = reshape( (1/sqNr)*fft2(reshape(X,sqNr,sqNr,L)) ,Nr,L );
    BhX_fast = Bhfast(X,Nr);
end;
toc;

max(abs(BhX(:)-BhX_fast(:)))
