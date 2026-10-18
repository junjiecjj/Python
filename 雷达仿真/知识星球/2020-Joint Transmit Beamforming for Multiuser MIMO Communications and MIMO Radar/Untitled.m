clc;
clear all;
close all;
warning off;
%% ------------- System Parameters ----------------
N = 32; % transmit signal block size 论文中是1024
M = 10; % M individual radar waveforms  M>K
K = 4; % K parallel communication symbol streams 2,4,6
Pt = 10^(0/10); % total_transmit_power
c=3e8; % speed of light
fc=3.2e9; % carrier frequency
lamda=c/fc; % wavelength
d=lamda/2; % adjacent antenna distance
wc = 1; % Lr2 weight
SNRdB = -2:2:12;
noise_variance = 0.01;
SINR_threshold = 12;
N_montecarlo = 10;

%% ------------- Radar Parameters -----------------
delta = pi/180; % 论文中是0.1度的分辨率
theta = -pi/2:delta:pi/2; % directional angle grids
theta_target = [-40*delta ,0 ,40*delta]; % three main lobe directions
P = length(theta_target); % length of target angle
beam_width = 10; % 10 degrees for each main lobe
l = ceil((theta_target+pi/2*ones(1,length(theta_target)))/(delta)+ones(1,length(theta_target)));
d_theta = zeros(length(theta),1);
for ii=1:length(theta_target)
    d_theta(l(ii)-(beam_width-1)/2:l(ii)+(beam_width-1)/2,1) = ones(beam_width,1); % given desired beampattern
end
a = ULA_steering_vector(M,theta);
a_theta_bar = ULA_steering_vector(M,theta_target);

%% ---------------- Beampattern ------------------
R_0 = zeros(M,M,N_montecarlo);
R__ZF = zeros(M,M,N_montecarlo);
R__SDR = zeros(M,M,N_montecarlo);
p = zeros(K,N_montecarlo);
for nn = 1:N_montecarlo
    H = (randn(K,M)+1i*randn(K,M))/sqrt(2); % instantaneous channel matrix
%     N_pbits = 2*K*N;
%     msg_bits = randi([0,1],1,N_pbits);
%     Y = reshape(QPSK_mapper(msg_bits),[K,N]);
%     X_orth = Orthogonal_Com_Rad( H',Y,Pt );
    R_0(:,:,nn) = waveform_design_radar_only_covmat( d_theta,M,P,a,a_theta_bar,theta,Pt);
    [R__ZF(:,:,nn),p(:,nn)] = waveform_design_ZF_covmat( d_theta,H,M,K,P,a,a_theta_bar,theta,SINR_threshold,noise_variance,Pt );
    [R__SDR(:,:,nn),r] = waveform_design_SDR_covmat( d_theta,H,M,K,P,a,a_theta_bar,theta,SINR_threshold,noise_variance,Pt );
    clc;
    disp(['Progress Beampattern- ',num2str(nn),'/',num2str(N_montecarlo)]);
end
R0 = mean(R_0,3);
R_ZF = mean(R__ZF,3);
R_SDR = mean(R__SDR,3);

%% --------------- Beampattern MSE ---------------- 
% allocate memory
MSE_ZF_K2 = zeros(1,18-10+1);
MSE_ZF_K4 = zeros(1,18-10+1);
MSE_ZF_K6 = zeros(1,18-10+1);
MSE_SDR_K2 = zeros(1,18-10+1);
MSE_SDR_K4 = zeros(1,18-10+1);
MSE_SDR_K6 = zeros(1,18-10+1);
INR_K2 = zeros(1,18-10+1);
INR_K4 = zeros(1,18-10+1);
INR_K6 = zeros(1,18-10+1);
SUM_RATE_ZF_K2 = zeros(1,18-10+1);
SUM_RATE_ZF_K4 = zeros(1,18-10+1);
SUM_RATE_ZF_K6 = zeros(1,18-10+1);
SUM_RATE_SDR_K2 = zeros(1,18-10+1);
SUM_RATE_SDR_K4 = zeros(1,18-10+1);
SUM_RATE_SDR_K6 = zeros(1,18-10+1);
% instantaneous channel Matrix for different K values
H_K2 = (randn(2,M)+1i*randn(2,M))/sqrt(2); %  channel matrix
H_K4 = (randn(4,M)+1i*randn(4,M))/sqrt(2); % instantaneous channel matrix
H_K6 = (randn(6,M)+1i*randn(6,M))/sqrt(2); % instantaneous channel matrix

for sinr_threshold=10:1:18
    R_0 = zeros(M,M,N_montecarlo);
    R__ZF_K2 = zeros(M,M,N_montecarlo);
    R__ZF_K4 = zeros(M,M,N_montecarlo);
    R__ZF_K6 = zeros(M,M,N_montecarlo);
    R__SDR_K2 = zeros(M,M,N_montecarlo);
    R__SDR_K4 = zeros(M,M,N_montecarlo);
    R__SDR_K6 = zeros(M,M,N_montecarlo);
    p_K2 = zeros(K,N_montecarlo);
    p_K4 = zeros(K,N_montecarlo);
    p_K6 = zeros(K,N_montecarlo);
    for nn = 1:N_montecarlo
        R_0(:,:,nn) = waveform_design_radar_only_covmat( d_theta,M,P,a,a_theta_bar,theta,Pt);
        [R__ZF_K2(:,:,nn),p_K2] = waveform_design_ZF_covmat( d_theta,H_K2,M,2,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__SDR_K2(:,:,nn),r_K2] = waveform_design_SDR_covmat( d_theta,H_K2,M,2,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__ZF_K4(:,:,nn),p_K4] = waveform_design_ZF_covmat( d_theta,H_K4,M,4,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__SDR_K4(:,:,nn),r_K4] = waveform_design_SDR_covmat( d_theta,H_K4,M,4,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__ZF_K6(:,:,nn),p_K6] = waveform_design_ZF_covmat( d_theta,H_K6,M,6,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__SDR_K6(:,:,nn),r_K6] = waveform_design_SDR_covmat( d_theta,H_K6,M,6,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
    end
    R0 = mean(R_0,3);
    R_ZF_K2 = mean(R__ZF_K2,3);
    R_ZF_K4 = mean(R__ZF_K4,3);
    R_ZF_K6 = mean(R__ZF_K6,3);
    R_SDR_K2 = mean(R__SDR_K2,3);
    R_SDR_K4 = mean(R__SDR_K4,3);
    R_SDR_K6 = mean(R__SDR_K6,3);
    pK2 = mean(p_K2,2);
    pK4 = mean(p_K4,2);
    pK6 = mean(p_K6,2);
    INR_K2(sinr_threshold-9) = INR_func(R_SDR_K2,H_K2,r_K2,M,2,noise_variance);
    INR_K4(sinr_threshold-9) = INR_func(R_SDR_K4,H_K4,r_K4,M,4,noise_variance);
    INR_K6(sinr_threshold-9) = INR_func(R_SDR_K6,H_K6,r_K6,M,6,noise_variance);
    SUM_RATE_SDR_K2(sinr_threshold-9) = SUM_RATE_func(R_SDR_K2,H_K2,r_K2,M,2,noise_variance);
    SUM_RATE_SDR_K4(sinr_threshold-9) = SUM_RATE_func(R_SDR_K4,H_K4,r_K4,M,4,noise_variance);
    SUM_RATE_SDR_K6(sinr_threshold-9) = SUM_RATE_func(R_SDR_K6,H_K6,r_K6,M,6,noise_variance);
    SUM_RATE_ZF_K2(sinr_threshold-9) = sum(log2(pK2/noise_variance));
    SUM_RATE_ZF_K4(sinr_threshold-9) = sum(log2(pK4/noise_variance));
    SUM_RATE_ZF_K6(sinr_threshold-9) = sum(log2(pK6/noise_variance));
    MSE_ZF_K2(sinr_threshold-9) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_ZF_K2*a)/real(trace(R_ZF_K2)),2).^2/length(d_theta);
    MSE_ZF_K4(sinr_threshold-9) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_ZF_K4*a)/real(trace(R_ZF_K4)),2).^2/length(d_theta);
    MSE_ZF_K6(sinr_threshold-9) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_ZF_K6*a)/real(trace(R_ZF_K6)),2).^2/length(d_theta);
    MSE_SDR_K2(sinr_threshold-9) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_SDR_K2*a)/real(trace(R_SDR_K2)),2).^2/length(d_theta);
    MSE_SDR_K4(sinr_threshold-9) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_SDR_K4*a)/real(trace(R_SDR_K4)),2).^2/length(d_theta);
    MSE_SDR_K6(sinr_threshold-9) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_SDR_K6*a)/real(trace(R_SDR_K6)),2).^2/length(d_theta);
    clc;
    disp(['Progress SINR_threshold- ',num2str(sinr_threshold),'/',num2str(18)]);
end

%% ----------------- Figures ----------------------
figure(1); % Beampattern versus direction
plot(theta*180/pi,10*log10(diag(a'*R0*a)/real(trace(R0))),'b-','LineWidth',1.5);hold on;
plot(theta*180/pi,10*log10(diag(a'*R_ZF*a)/real(trace(R_ZF))),'r.-','LineWidth',1.5);hold on;
plot(theta*180/pi,10*log10(diag(a'*R_SDR*a)/real(trace(R_SDR))),'g.-','LineWidth',1.5);hold on;
grid on;
xlim([-90,90]);
xlabel('\theta°');
ylabel('Transmit Beampattern (dB)');
legend('Radar only','ZF','SDR');
title(strcat('SINR threshold=',num2str(SINR_threshold),'dB, K=',num2str(K)));

%%
SINR_threshold = 10:1:18;
figure(2); % Beampattern MSE versus SINR_threshold
semilogy(SINR_threshold,MSE_ZF_K2,'bx-','LineWidth',1.5,'MarkerSize',8);hold on;
semilogy(SINR_threshold,MSE_ZF_K4,'go-','LineWidth',1.5,'MarkerSize',8);hold on;
semilogy(SINR_threshold,MSE_ZF_K6,'r^-','LineWidth',1.5,'MarkerSize',8);hold on;
semilogy(SINR_threshold,MSE_SDR_K2,'b*-','LineWidth',1.5,'MarkerSize',8);hold on;
semilogy(SINR_threshold,MSE_SDR_K4,'g--','LineWidth',1.5,'MarkerSize',8);hold on;
semilogy(SINR_threshold,MSE_SDR_K6,'r+--','LineWidth',1.5,'MarkerSize',8);
grid on;
xlabel('SINR threshold (dB)');
ylabel('Beampattern MSE');
legend('ZF,K=2','ZF,K=4','ZF,K=6','SDR,K=2','SDR,K=4','SDR,K=6');
% figure(3); % Sum Rate versus SINR_threshold
% plot(SINR_threshold,SUM_RATE_ZF_K2,'bx-','LineWidth',1.5,'MarkerSize',8);hold on;
% plot(SINR_threshold,SUM_RATE_ZF_K4,'go-','LineWidth',1.5,'MarkerSize',8);hold on;
% plot(SINR_threshold,SUM_RATE_ZF_K6,'r^-','LineWidth',1.5,'MarkerSize',8);hold on;
% plot(SINR_threshold,SUM_RATE_SDR_K2,'b*-','LineWidth',1.5,'MarkerSize',8);hold on;
% plot(SINR_threshold,SUM_RATE_SDR_K4,'g--','LineWidth',1.5,'MarkerSize',8);hold on;
% plot(SINR_threshold,SUM_RATE_SDR_K6,'r+--','LineWidth',1.5,'MarkerSize',8);
% grid on;
% xlabel('SINR threshold (dB)');
% ylabel('Sum Rate (bps/Hz)');
% legend('ZF,K=2','ZF,K=4','ZF,K=6','SDR,K=2','SDR,K=4','SDR,K=6');
% figure(4); % Interference-noise-ratio versus SINR_threshold
% semilogy(SINR_threshold,INR_K2,'bx-','LineWidth',1.5,'MarkerSize',8);hold on;
% semilogy(SINR_threshold,INR_K4,'go-','LineWidth',1.5,'MarkerSize',8);hold on;
% semilogy(SINR_threshold,INR_K6,'r^-','LineWidth',1.5,'MarkerSize',8);hold on;
% grid on;
% xlabel('SINR threshold (dB)');
% ylabel('Interference-noise-ratio dB');
% legend('ZF,K=2','ZF,K=4','ZF,K=6','SDR,K=2','SDR,K=4','SDR,K=6');