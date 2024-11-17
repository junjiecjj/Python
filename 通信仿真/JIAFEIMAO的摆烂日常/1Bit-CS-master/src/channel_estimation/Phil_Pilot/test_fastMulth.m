clear;
%% under revision

Nt=64; % number of Tx antennas
L=16; % number of delays in channel
P=4.5; % Nyquist oversampling rate
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
% t = exp((1i*pi/N)*[0:N-1].*([0:N-1]+1));
t = randn(1, N) + 1j*rand(1,N);

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
in = randn(Nr,N) + 1i*randn(Nr,N);


% X = reshape(in,Nr,[]);
% XX = fft( bsxfun(@times,ifft(X',N, 1), tf), N, 1)';

% TT = reshape(TJ, [Nt, L, N]);
% TTT = permute(TT, [2 1 3]);
% Tperm = reshape(TTT, [Nt*L, N]);
%
% temp1 = (Tperm * X')';
% temp2 = X * Tperm';
%
% temp3 = (dftmtx(N) * diag(tf) * dftmtx(N)'/N * X')';
%
% norm(temp1-temp3)
%
% norm(temp1 - XX);

% XX = reshape(X,[Nr,Nt,L]);
% XXp = permute(XX,[1,3,2]);
% Xperm = reshape(XXp,[Nr,Nt*L]);

% % create noiseless outputs
% Z = Br' * reshape(in, Nr,[]) * TJ' * kron(eye(L),Bt) ;

% % test fast method
% %Ztest = Bfast( X * kron(eye(L),ctranspose(Bt)) * TJ , Nr);
% %Ztest = Bfast( Xperm * kron(ctranspose(Bt),eye(L)) * TJpermt.' , Nr);
% %Ztest = Bfast( (TJpermt * kron(ctranspose(Bt),eye(L)) * Xperm.').' , Nr);
% %Ztest = Bfast( ( TJpermt * BhKronIfast(Xperm.',Nt) ).' , Nr);
% %Ztest = Bfast( ( ifft( (tf*ones(1,Nr)) .* fft(BhKronIfast(Xperm.',Nt),N) ) ).' , Nr);
% Ztest = Bfast( ifft( bsxfun(@times,tf,fft(BhKronIfast(Xperm.',Nt),N)) ).' , Nr);
% max(abs(Z(:)-Ztest(:)))
%
% time original and fast method
tic
for j=1:100
    Z = Br' * reshape(in, Nr,[]) * TJ' * kron(eye(L),Bt) ;
end
toc

tic
for j=1:100
    tf = conj( fft((T(1,:)),N,2)); %% pay attention to the 'conj' here. I made a mistake before.
    
    X = reshape(in,Nr,[]);
    
    % XX = Br' * ifft( bsxfun(@times,tf,fft(X,N, 2)), N, 2);
    % Fast implementation
    XX = ifft( bsxfun(@times,tf,fft(X,N, 2)), N, 2);
    XX = Bhfast( XX(1:Nr, 1:Nt*L) , Nr);
    
    XXX = reshape(XX, Nr, L, []);
    XXXp = permute(XXX, [1 3 2]);
    XXXperm = reshape(XXXp, Nr, []);
    
    % out = reshape(XXXperm * kron(eye(L), Bt), [], 1);
    % Fast implementation
    Ztest = reshape( X_IKronBfast(XXXperm, Nt), [], 1 );
end
toc

norm(Z(:)-Ztest(:))