clc; clear all; close all;
% SIGNAL MODEL OF OTFS
M = 32; N = 32;                 %M-> no. of subcarriers/delay bins ; N-> no. of symbols/doppler bins
F_M = 1/sqrt(M)*dftmtx(M);      %disrete fourier transform matrix (normalised)   
F_N = 1/sqrt(N)*dftmtx(N);      %forms the grid
Ptx = eye(M);
delta_f = 15e3;                 %subcarrier BW
T = 1/delta_f;                  %OFDM symbol Duration

                   
X_DD = zeros(M,N);              %Delay Doppler Grid
X_DD(3, 3) = 1;                 %impulse in DD Domain
X_TF = F_M*X_DD*F_N';            
S = Ptx*F_M'*X_TF;
s = reshape(S,M*N,1);




% figure()
subplot(4,2,[1,2,3,4])
bar3(X_DD);
axis tight;
xlabel('Doppler'); 
ylabel('Delay');
title('Basis function in DD-domain');

% figure()
subplot(4,2,5)
surf(real(X_TF));
axis tight;
xlabel('Time'); 
ylabel('Subcarrier');
title('Basis function in TF-domain (Real)');

% figure()
subplot(4,2,6)
surf(imag(X_TF));
axis tight;
xlabel('Time'); 
ylabel('Subcarrier');
title('Basis function in TF-domain (Imag)');

% figure()
subplot(4,2,7)
plot((0:length(s)-1)*T/M,real(s));
axis tight;
ylim([-0.5 0.5]);
xlabel('Time');
title('Basis function in time-domain (Real)');

% figure()
subplot(4,2,8)
plot((0:length(s)-1)*T/M,imag(s));
axis tight;
ylim([-0.5 0.5]);
xlabel('Time');
title('Basis function in time-domain (Imag)');

