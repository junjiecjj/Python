% Tests fast implementation of B*X where B=kron(DFT,DFT) and X is a matrix
% JMo: When x is a vector, the results show that the fast operations actully are slower than the
% direct mutliplication. :-(. However, when X is a matrix with multiple
% columns, the fast implement saves a lot of time. :-)

Nr = 64;

% explicit representation of 2D-ULA matrix
sqNr = sqrt(Nr);
F = (1/sqrt(sqNr))*dftmtx(sqNr);
B = kron(F,F);

%% test fast multiplication by vector
x = randn(Nr,1)+1i*randn(Nr,1);
tic;
Bx = B*x;
toc;

tic;
%Bx_fast = reshape( (1/sqNr)*fft2(reshape(x,sqNr,sqNr)) ,Nr,1 );
Bx_fast = Bfast(x,Nr);
toc;

max(abs(B*x-Bx_fast))

%% test fast multiplication by matrix
L = 2000;
X = randn(Nr,L)+1i*randn(Nr,L);

tic;
for i=1:1:1e3
    BX = B*X;
end;
toc;

tic;
for i=1:1:1e3
    %BX_fast = reshape( (1/sqNr)*fft2(reshape(X,sqNr,sqNr,L)) ,Nr,L );
    BX_fast = Bfast(X,Nr);
end;
toc;

max(abs(BX(:)-BX_fast(:)))
