% Clear command window, workspace variables, and close all figures
clc; 
clear all; 
close all;

% Define Eb values in dB
EbdB = 2;

% Convert Eb values from dB to linear scale
Eb = 10.^(EbdB/10);

% Define Noise Power
No = 1;

% Calculate Signal-to-Noise Ratio (SNR) in linear scale
SNR = 2*Eb/No;

% Convert SNR to dB scale
SNRdB = 10*log10(SNR);

% Define matrix dimensions and parameters
M = 32; 
N = 16;
Ptx = eye(M); 
Prx = eye(M);
nTaps = 5;
DelayTaps = [5 1 0 3 4];
DopplerTaps = [0 3 2 3 4];
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
xhatMMSE = inv(Heff'*Heff + eye(M*N)/Eb)*Heff'*y_DD;
DecodedBits_MMSE = (real(xhatMMSE) >= 0);
DecodedBits_MMSE_reshaped = reshape(DecodedBits_MMSE,M,N);
BER_MMSE_Map = (DecodedBits_MMSE_reshaped ~= XddBits);
BER_MMSE = sum(DecodedBits_MMSE ~= reshape(XddBits,M*N,1));

% Zero Forcing (ZF) Equalization
xhatZF = pinv(Heff)*y_DD;
DecodedBits_ZF = (real(xhatZF) >= 0);
DecodedBits_ZF_reshaped = reshape(DecodedBits_ZF,M,N);
BER_ZF_MAP = (DecodedBits_ZF_reshaped ~= XddBits);
BER_ZF = sum(DecodedBits_ZF ~= reshape(XddBits,M*N,1));

% Average BER over iterations and symbols
BER_MMSE = BER_MMSE/M/N;
BER_ZF = BER_ZF/M/N;

%% Plots
plt1 = zeros(50,1);
basis_plt1 = [1,2,3,4,5];
for i = 1:10
    plt1(i*5-4) = 15*(i-1) + basis_plt1(1);
    plt1(i*5-3) = 15*(i-1) +basis_plt1(2);
    plt1(i*5-2) = 15*(i-1) +basis_plt1(3);
    plt1(i*5-1) = 15*(i-1) +basis_plt1(4);
    plt1(i*5) = 15*(i-1) +basis_plt1(5);
end
plt2 = plt1 +5;
plt3 = plt1 + 10;
plt4 = zeros(15,1);
basis_plt2 = [1,2,3];
for i = 1:5
    plt4(i*3-2) = 15*(i+9) + basis_plt2(1);
    plt4(i*3-1) = 15*(i+9) +basis_plt2(2);
    plt4(i*3) = 15*(i+9) +basis_plt2(3);
end
plt5 = plt4+3;
plt6 = zeros(18,1);
basis_plt3 = [7,8,9,10,11,12,13,14,15];
for i = 1:2
    plt6(i*9-8) = 15*(i+9) + basis_plt3(1);
    plt6(i*9-7) = 15*(i+9) +basis_plt3(2);
    plt6(i*9-6) = 15*(i+9) +basis_plt3(3);
    plt6(i*9-5) = 15*(i+9) +basis_plt3(4);
    plt6(i*9-4) = 15*(i+9) +basis_plt3(5);
    plt6(i*9-3) = 15*(i+9) +basis_plt3(6);
    plt6(i*9-2) = 15*(i+9) +basis_plt3(7);
    plt6(i*9-1) = 15*(i+9) +basis_plt3(8);
    plt6(i*9) = 15*(i+9) +basis_plt3(9);
end

plt7 = zeros(18,1);
for i = 1:2
    plt7(i*9-8) = 15*(i+12) + basis_plt3(1);
    plt7(i*9-7) = 15*(i+12) +basis_plt3(2);
    plt7(i*9-6) = 15*(i+12) +basis_plt3(3);
    plt7(i*9-5) = 15*(i+12) +basis_plt3(4);
    plt7(i*9-4) = 15*(i+12) +basis_plt3(5);
    plt7(i*9-3) = 15*(i+12) +basis_plt3(6);
    plt7(i*9-2) = 15*(i+12) +basis_plt3(7);
    plt7(i*9-1) = 15*(i+12) +basis_plt3(8);
    plt7(i*9) = 15*(i+12) +basis_plt3(9);
end

txt1 = 'SNR = %d, EbdB';



% figure() 1
subplot(15,15,plt1);
bar3(XddBits);
axis tight;
xlabel('Doppler'); 
ylabel('Delay');
title('Transmitted');

% figure() 2
subplot(15,15,plt2);
bar3(reshape(DecodedBits_MMSE,M,N));
hold on
bar3(BER_MMSE_Map,'r');
axis tight;
xlabel('Doppler'); 
ylabel('Delay');
title('MMSE Reciever');

% figure() 3
subplot(15,15,plt3);
bar3(reshape(DecodedBits_ZF,M,N));
hold on
bar3(BER_ZF_MAP,'r');
axis tight;
xlabel('Doppler'); 
ylabel('Delay');
title('ZF Reciever');

% figure() 4
subplot(15,15,plt4);
bar3(X_DD);
axis tight;
xlabel('Doppler'); 
ylabel('Delay');
title('Symbols map Transmitted');

% figure() 5
subplot(15,15,plt5);
bar3(Y_DD);
axis tight;
xlabel('Doppler'); 
ylabel('Delay');
title('Symbols map Recieved');

% figure() 6
subplot(15,15,plt6);
surf(real(X_TF));
axis tight;
xlabel('Time'); 
ylabel('Subcarrier');
title('Transmitted TF-domain (Real)');

% figure() 7
subplot(15,15,plt7);
surf(real(Y_TF));
axis tight;
xlabel('Time'); 
ylabel('Subcarrier');
title('Recieved TF-domain (Real)');
