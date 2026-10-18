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
N_montecarlo = 5;

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
