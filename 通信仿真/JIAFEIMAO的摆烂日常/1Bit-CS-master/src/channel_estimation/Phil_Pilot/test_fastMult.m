Nt=64; % number of Tx antennas
L=16; % number of delays in channel
P=5; % Nyquist oversampling rate
N=Nt*L*P; % training length
Nr=64; % number of Rx antennas

% create B=kron(F,F) matrices
sqNr = sqrt(Nr);
Fr = (1/sqrt(sqNr))*dftmtx(sqNr);
Br = kron(Fr,Fr);
sqNt = sqrt(Nt);
Ft = (1/sqrt(sqNt))*dftmtx(sqNt);
Bt = kron(Ft,Ft);

% create training matrix T of size Nt-by-N
% random BPSK
%t = sign(randn(1,N));

% flat in frequency, all pass signal
%t = (1/sqrt(N))*fft(sign(exp(1i*2*pi*rand(N,1))))';

% chu sequence
t = exp((1i*pi/N)*[0:N-1].*([0:N-1]+1));

% time-domain spike
% t = [sqrt(N),zeros(1,N-1)];

T = zeros(Nt,N);
for i=0:Nt-1
    T(i+1,:) = [t(N-i*L+1:N),t(1:N-i*L)];
end
tf = fft(t.'); % frequency-domain version of pulse

% create Nt*L-by-N block-convolution matrix
TJ = T;
for l=1:L-1
    TJ = [TJ;[T(:,N-l+1:N),T(:,1:N-l)]];
end

% create Nt*L-by-N circulant matrix with first column t.'
TJpermt = toeplitz(t.',[t(1),t(N:-1:N-(Nt*L)+2)]);

% create channel coefficient matrix
X = randn(Nr,Nt*L) + 1i*randn(Nr,Nt*L);
XX = reshape(X,[Nr,Nt,L]);
XXp = permute(XX,[1,3,2]);
Xperm = reshape(XXp,[Nr,Nt*L]);

% create noiseless outputs
Z = Br * X * kron(eye(L),ctranspose(Bt)) * TJ;

% test fast method
%Ztest = Bfast( X * kron(eye(L),ctranspose(Bt)) * TJ , Nr);
%Ztest = Bfast( Xperm * kron(ctranspose(Bt),eye(L)) * TJpermt.' , Nr);
%Ztest = Bfast( (TJpermt * kron(ctranspose(Bt),eye(L)) * Xperm.').' , Nr);
%Ztest = Bfast( ( TJpermt * BhKronIfast(Xperm.',Nt) ).' , Nr);
%Ztest = Bfast( ( ifft( (tf*ones(1,Nr)) .* fft(BhKronIfast(Xperm.',Nt),N) ) ).' , Nr);
Ztest = Bfast( ifft( bsxfun(@times,tf,fft(BhKronIfast(Xperm.',Nt),N)) ).' , Nr);
max(abs(Z(:)-Ztest(:)))

% time original and fast method

% tic
% for j=1:100
%     L_Bt = kron(eye(L),ctranspose(Bt));
% end;
% time_L_Bt = toc
% 
% tic
% for j=1:100,
%     Z = Br * X * L_Bt * TJ;
% end
% toc

tic
for j=1:100,
    Z = Br * X * kron(eye(L),ctranspose(Bt)) * TJ;
end
toc

tic
for j=1:100
    Ztest = Bfast( ifft( bsxfun(@times,tf,fft(BhKronIfast(Xperm.',Nt),N)) ).' , Nr);
end
toc

