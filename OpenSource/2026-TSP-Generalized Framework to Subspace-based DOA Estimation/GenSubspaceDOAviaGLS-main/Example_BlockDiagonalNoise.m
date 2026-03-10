clear;
close all;
M=20;                    % Number of elements of the array
degsais=[34 36 -10];     % True DOAs in Degree
psais = degsais*pi/180;  % True DOAs in Radian
K = length(psais);       % Number of sources
SNR_db = 15;             % SNR in dB
SNR = 10.^(SNR_db/10);
N = 100;                  % Number of snapshots


Q1=zeros(4,4);
sig1=5;
kisi1=0.7;

Q2=zeros(3,3);
sig2=3;
kisi2=1;

Q3=zeros(2,2);
sig3=1;
kisi3=0.5;

for ii=1:4
    for jj=1:4
        Q1(ii,jj)=sig1*exp(-((ii-jj)^2)*kisi1);
    end
end


for ii=1:3
    for jj=1:3
        Q2(ii,jj)=sig2*exp(-((ii-jj)^2)*kisi2);
    end
end


for ii=1:2
    for jj=1:2
        Q3(ii,jj)=sig3*exp(-((ii-jj)^2)*kisi3);
    end
end



Qv=blkdiag(Q1,diag([10 3 2 3 1]),Q2,diag([7 5 4 7 1 1]),Q3);    % Actual Noise Covariance Matrix
eps = 1e-4;
%% Generating Received Signal
suminvQv = sum(1./diag(Qv));
sigmasq = (M/suminvQv)*SNR;    % Power of each source          

A = exp(-1j*pi*(0:M-1)'*sin(psais));     % Steering Matrix
S = sqrt(sigmasq/2)*(randn(K,N)+1j*randn(K,N));     % Source Signal
Noise = sqrt(1/2)*(Qv^(0.5))*(randn(M,N)+1j*randn(M,N));    % Noise Matrix
xmt = (A*S)+Noise;                  % Received Signal

R=(xmt*xmt')/N;
%% Estimating Noise Covariance Matrix
[Q] = BDISB(R,K,M,eps);

%% Proposed Methods 
[PM_DOA_degree,PM_DOA_radian] = Proposed(R,K,M,5,xmt,[ M-1 :M ],Q);            % Proposed Method
[FB_PM_DOA_degree,FB_PM_DOA_radian] = FB_Proposed(R,K,M,5,xmt,[ M-1 :M ],Q,N);   % FB Proposed Method