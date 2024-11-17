clc;
clear;
A = randn(5,3);
B = [ 1 2 3 4 5;
5 1 2 3 4;
4 5 1 2 3;
3 4 5 1 2;
2 3 4 5 1]*1j;

C = [A zeros(5,2)] * B

af = ifft(A, 5, 2);
bf = fft([1 5 4 3 2]*1j);

D = fft(bsxfun(@times,af, bf), 5,2)

Nt = 16;
Nr = 4;
N = 256;
L = 16;
if mod(N,2) == 0
    s = exp((1i*pi/N)*(0:N-1).^2);  % even length
else
    s = exp((1i*pi/N)*(0:N-1).*((0:N-1)+1)); % odd length
end;
T0 = zeros(Nt,N);
for i=0:Nt-1
    T0(i+1,:) = [s(N-i*L+1:N),s(1:N-i*L)];
end
T_train = T0;


Nta = 4;  % azimuth departure angle
Nte = 4;  % elevation departure angle
Bt = 1/sqrt(Nt) * kron( dftmtx(Nta), dftmtx(Nte) );

V_train = Bt'* T_train(1:Nt,:);
for ii=0:1:L-1
    V( ii*Nt+1: (ii+1)*Nt, :) = circshift(V_train,ii,2);
end

VV = reshape(V,[Nt,L,N]);
VVp = permute(VV,[2,1,3]);
Vperm = reshape(VVp,[Nt*L, N]);

Vperm(1,2:end)-Vperm(end, 1:end-1)

in = randn(1, Nr*Nt*L) + 1j* randn(1, Nr*Nt*L);
X = reshape(in, Nr, []);
XV = X*V;


vf = fft( V(1,:),N,2) ;

XX = reshape(in,[Nr,Nt,L]);
XXp = permute(XX,[1,3,2]);
Xperm = reshape(XXp,[Nr,Nt*L]);

% Xperm_Vperm = Xperm*Vperm;
% norm(XV(:) - Xperm_Vperm(:))

temp_FFT = ifft( bsxfun(@times,vf, fft(Xperm,N,2)), N, 2);

norm(XV(:) - temp_FFT(:))