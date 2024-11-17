clc;
clear;

Nt=64;
L=16;
P=2;
N=Nt*L*P;
N = N - 1

% iid Gaussian
T = (1/sqrt(2))*(randn(Nt*L,N)+ 1i*randn(Nt*L,N)); 
svd_T_iid = svd(T);
peak_to_avg_iid = max(svd_T_iid.^2)/mean(svd_T_iid.^2)

% block-conv via iid Gaussian 
T0 = (1/sqrt(2))*(randn(Nt,N)+ 1i*randn(Nt,N)); 
T = T0;
for l=1:L-1
  T = [T;[T0(:,N-l+1:N),T0(:,1:N-l)]];
end
svd_Tc_iid = svd(T);
peak_to_avg_ciid = max(svd_Tc_iid.^2)/mean(svd_Tc_iid.^2)

% block-conv via iid rademacher 
T0 = sign(randn(Nt,N)); 
T = T0;
for l=1:L-1
  T = [T;[T0(:,N-l+1:N),T0(:,1:N-l)]];
end
svd_Tc_rad = svd(T);
peak_to_avg_crad = max(svd_Tc_rad.^2)/mean(svd_Tc_rad.^2)

%% block-conv via randomly-sampled twice hadamard 
%rad = (1/sqrt(N))*sign(randn(1,N)); 
%H = hadamard(N); T0 = H(randperm(N,Nt),:)*diag(rad)*H;
%T = T0;
%for l=1:L-1
%  T = [T;[T0(:,N-l+1:N),T0(:,1:N-l)]];
%end
%svd_Tc_had2 = svd(T);
%peak_to_avg_chad2 = max(svd_Tc_had2.^2)/mean(svd_Tc_had2.^2)

%% block-conv via randomly-sampled hadamard 
%H = hadamard(N); T0 = H(randperm(N,Nt),:);
%T = T0;
%for l=1:L-1
%  T = [T;[T0(:,N-l+1:N),T0(:,1:N-l)]];
%end
%svd_Tc_had = svd(T);
%peak_to_avg_chad = max(svd_Tc_had.^2)/mean(svd_Tc_had.^2)

%% block-conv via randomly-sampled DFT 
%F = dftmtx(N); T0 = F(randperm(N,Nt),:);
%T = T0;
%for l=1:L-1
%  T = [T;[T0(:,N-l+1:N),T0(:,1:N-l)]];
%end
%svd_Tc_dft = svd(T);
%peak_to_avg_cdft = max(svd_Tc_dft.^2)/mean(svd_Tc_dft.^2)

% block-conv via shifted random BPSK
s = sign(randn(1,N)); % random BPSK
T0 = zeros(Nt,N);
for i=0:Nt-1
  T0(i+1,:) = [s(N-i*L+1:N),s(1:N-i*L)];
end
T = T0;
for l=1:L-1
  T = [T;[T0(:,N-l+1:N),T0(:,1:N-l)]];
end
svd_Tc_bpsk = svd(T);
peak_to_avg_cbpsk = max(svd_Tc_bpsk.^2)/mean(svd_Tc_bpsk.^2)

% block-conv via shifted chu
% s = exp((1i*pi/N)*[0:N-1].*([0:N-1]+1));% chu sequence, incorrect

if mod(N,2) == 0
    s = exp((1i*pi/N)*[0:N-1].^2);  % even length
else
    s = exp((1i*pi/N)*[0:N-1].*([0:N-1]+1)); % odd length
end;
    
T0 = zeros(Nt,N);
for i=0:Nt-1
  T0(i+1,:) = [s(N-i*L+1:N),s(1:N-i*L)];
end
T = T0;
for l=1:L-1
  T = [T;[T0(:,N-l+1:N),T0(:,1:N-l)]];
end
svd_Tc_chu = svd(T);
peak_to_avg_cchu = max(svd_Tc_chu.^2)/mean(svd_Tc_chu.^2)

% block-conv via shifted allpass
s = (1/sqrt(N))*fft(sign(exp(1i*2*pi*rand(N,1))))'; % random allpass
%s = [sqrt(N),zeros(1,N-1)]; % time-domain spike
T0 = zeros(Nt,N);
for i=0:Nt-1
  T0(i+1,:) = [s(N-i*L+1:N),s(1:N-i*L)];
end
T = T0;
for l=1:L-1
  T = [T;[T0(:,N-l+1:N),T0(:,1:N-l)]];
end
svd_Tc_allpass = svd(T);
peak_to_avg_callpass = max(svd_Tc_allpass.^2)/mean(svd_Tc_allpass.^2)


% plot svds
plot([1:Nt*L],svd_T_iid,...
     [1:Nt*L],svd_Tc_bpsk,...
     [1:Nt*L],svd_Tc_chu,...
     [1:Nt*L],svd_Tc_allpass);
     %[1:Nt*L],svd_Tc_iid,...
     %[1:Nt*L],svd_Tc_rad,'--',...
ylabel('singular values of T')
legend('iid gaussian',...
       'blk-conv-mtx from L-shifted BPSK',...
       'blk-conv-mtx from L-shifted chu',...
       'blk-conv-mtx from L-shifted allpass sequence')
       %'blk-conv-mtx from iid gaussian',...
       %'blk-conv-mtx from iid rademacher',...
title('singular value spectrum for different T matrices')

