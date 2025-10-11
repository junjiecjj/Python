% Clear command window, workspace variables, and close all figures
clc; 
clear all; 
close all;

% Define Eb values in dB
EbdB = 3;

% Convert Eb values from dB to linear scale
Eb = 10.^(EbdB/10);

% Define Noise Power
No = 1;

% Calculate Signal-to-Noise Ratio (SNR) in linear scale
SNR = 2*Eb/No;

% Convert SNR to dB scale
SNRdB = 10*log10(SNR);

% Define matrix dimensions and parameters
M = 16; 
N = 8;
Ptx = eye(M); 
Prx = eye(M);
nTaps = 4;
DelayTaps = [0 1 2 3];
DopplerTaps = [0 1 2 3];
Ncp = max(DelayTaps);


% Precompute matrices for transformation
F_M = 1/sqrt(M)*dftmtx(M);
F_N = 1/sqrt(N)*dftmtx(N);

% Generate random bits for transmission
XddBits = randi([0,1],M,N);

% Generate random channel taps
h = sqrt(1/2)*(randn(1,nTaps)+ 1j*randn(1,nTaps));

% Construct effective channel matrix
Hmat = zeros(M*N,M*N);
omega = exp(1j*2*pi/(M*N));
for tx = 1:nTaps
    Hmat = Hmat + h(tx)*circshift(eye(M*N),DelayTaps(tx))*...
        (diag(omega.^((0:M*N-1)*DopplerTaps(tx))));
end
Heff = kron(F_N,Prx)*Hmat*kron(F_N',Ptx);
    
% Generate Complex Noise
ChNoise = sqrt(No/2)*(randn(1,M*N) + 1j*randn(1,M*N));
    
% Generate modulated symbols
X_DD = sqrt(Eb)*(2*XddBits-1); 
X_TF = F_M*X_DD*F_N';
S_mat = Ptx*F_M'*X_TF;        
TxSamples = reshape(S_mat,M*N,1).';
TxSamplesCP = [TxSamples(M*N-Ncp+1:M*N) TxSamples];

% Channel filtering
RxsamplesCP = 0;
for tx = 1:nTaps
    Doppler = exp(1j*2*pi/M*(-Ncp:M*N-1)*DopplerTaps(tx)/N);
    RxsamplesCP = RxsamplesCP + h(tx)*circshift(TxSamplesCP.*Doppler,[1, DelayTaps(tx)]);
end

% Remove cyclic prefix
Rxsamples = RxsamplesCP(Ncp+1:M*N+Ncp) + ChNoise;
R_mat = reshape(Rxsamples.',M, N) ;
Y_TF = F_M*Prx*R_mat;
Y_DD = F_M'*Y_TF*F_N;
y_DD = reshape(Y_DD, M*N, 1) ;

% MMSE Equalization
xhatMMSE = pinv(Heff'*Heff + eye(M*N)/Eb)*Heff'*y_DD;
DecodedBits_MMSE = (real(xhatMMSE) >= 0);
DecodedBits_MMSE_reshaped = reshape(DecodedBits_MMSE,M,N);
BER_MMSE_Map = (DecodedBits_MMSE_reshaped ~= XddBits);
BER_MMSE = sum(DecodedBits_MMSE ~= reshape(XddBits,M*N,1));

[afmag, delay_af, doppler_af] = ambgfun(Rxsamples, 1000000, 10000);
[nlfmaf_delay, delay] = ambgfun(Rxsamples, 1000000, 10000,'Cut','Doppler');
figure()
% contour(delay,doppler,afmag,'ShowText','on');
mesh(delay_af, doppler_af,afmag);

% Average BER over iterations and symbols
BER_MMSE = BER_MMSE/M/N;

%% Plots
figure()
contour(delay_af,doppler_af,afmag)

figure()
ambgfun(TxSamples, 1000000, 10000,'Cut','Doppler','CutVal',5)

 
figure()
plot(delay*1e6,mag2db(nlfmaf_delay))