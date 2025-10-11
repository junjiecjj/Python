% Clear command window, workspace variables, and close all figures
clc; 
clear all; 
close all;

% Define Eb values in dB
EbdB = -10:2:10;

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
DelayTaps = [0 1 2 3 4];
DopplerTaps = [0 1 2 3 4];
Ncp = max(DelayTaps);

% Initialize arrays to store Bit Error Rate (BER) for different methods
BER_MMSE = zeros(length(Eb),1);
BER_ZF = zeros(length(Eb),1);

% Number of iterations for Monte Carlo simulation
ITER = 10;

% Precompute matrices for transformation
F_M = 1/sqrt(M)*dftmtx(M);
F_N = 1/sqrt(N)*dftmtx(N);

% Main loop for Monte Carlo simulation
for ite = 1:ITER
    ite
    
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
    
    % Loop over different Eb/N0 values
    for ix = 1:length(Eb)
        % Generate modulated symbols
        X_DD = sqrt(Eb(ix))*(2*XddBits-1); 
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
        xhatMMSE = inv(Heff'*Heff + eye(M*N)/Eb(ix))*Heff'*y_DD;
        DecodedBits = (real(xhatMMSE) >= 0);
        BER_MMSE(ix) = BER_MMSE(ix) + sum(DecodedBits ~= reshape(XddBits,M*N,1));
        
        % Zero Forcing (ZF) Equalization
        xhatZF = pinv(Heff)*y_DD;
        DecodedBits = (real(xhatZF) >= 0);
        BER_ZF(ix) = BER_ZF(ix) + sum(DecodedBits ~= reshape(XddBits,M*N,1));
    end
end

% Average BER over iterations and symbols
BER_MMSE = BER_MMSE/M/N/ITER;
BER_ZF = BER_ZF/M/N/ITER;

% Plot BER versus SNR
semilogy(EbdB,BER_MMSE,'b-s','linewidth',3.0,'MarkerFaceColor','b','MarkerSize',9.0);
hold on; 
grid on; 
semilogy(EbdB,BER_ZF,'r-o','linewidth',3.0,'MarkerFaceColor','r','MarkerSize',9.0);
axis tight;
legend('MMSE','ZF');
title('OTFS BER v/s SNR');
xlabel('SNR(dB)'); 
ylabel('BER');
