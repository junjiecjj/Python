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
SINR_threshold_lower = 10;
SINR_threshold_upper = 18;
SINR_step = 2;
N_montecarlo = 2;

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

%% --------------- Beampattern MSE ---------------- 
% allocate memory
MSE_ZF_K2 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
MSE_ZF_K4 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
MSE_ZF_K6 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
MSE_SDR_K2 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
MSE_SDR_K4 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
MSE_SDR_K6 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
INR_K2 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
INR_K4 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
INR_K6 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
SUM_RATE_ZF_K2 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
SUM_RATE_ZF_K4 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
SUM_RATE_ZF_K6 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
SUM_RATE_SDR_K2 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
SUM_RATE_SDR_K4 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
SUM_RATE_SDR_K6 = zeros(1,(SINR_threshold_upper-SINR_threshold_lower)/SINR_step+1);
% instantaneous channel Matrix for different K values
H_K2 = (randn(2,M)+1i*randn(2,M))/sqrt(2); % K=2
H_K4 = (randn(4,M)+1i*randn(4,M))/sqrt(2); % K=4
H_K6 = (randn(6,M)+1i*randn(6,M))/sqrt(2); % K=6

for sinr_threshold = SINR_threshold_lower:SINR_step:SINR_threshold_upper
    % allocate memory
    R_0 = zeros(M,M,N_montecarlo);
    R__ZF_K2 = zeros(M,M,N_montecarlo);
    R__ZF_K4 = zeros(M,M,N_montecarlo);
    R__ZF_K6 = zeros(M,M,N_montecarlo);
    R__SDR_K2 = zeros(M,M,N_montecarlo);
    R__SDR_K4 = zeros(M,M,N_montecarlo);
    R__SDR_K6 = zeros(M,M,N_montecarlo);
    p_K2 = zeros(2,N_montecarlo);
    p_K4 = zeros(4,N_montecarlo);
    p_K6 = zeros(6,N_montecarlo);
    W_ZF_K2 = zeros(M,M+2,N_montecarlo);
    W_ZF_K4 = zeros(M,M+4,N_montecarlo);
    W_ZF_K6 = zeros(M,M+6,N_montecarlo);
    W_SDR_K2 = zeros(M,M+2,N_montecarlo);
    W_SDR_K4 = zeros(M,M+4,N_montecarlo);
    W_SDR_K6 = zeros(M,M+6,N_montecarlo);
    for nn = 1:N_montecarlo
        R_0(:,:,nn) = waveform_design_radar_only_covmat( d_theta,M,P,a,a_theta_bar,theta,Pt);
        [R__ZF_K2(:,:,nn),p_K2(:,nn),W_ZF_K2(:,:,nn)] = waveform_design_ZF_covmat( d_theta,H_K2,M,2,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__SDR_K2(:,:,nn),r_K2,W_SDR_K2(:,:,nn)] = waveform_design_SDR_covmat( d_theta,H_K2,M,2,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__ZF_K4(:,:,nn),p_K4(:,nn),W_ZF_K4(:,:,nn)] = waveform_design_ZF_covmat( d_theta,H_K4,M,4,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__SDR_K4(:,:,nn),r_K4,W_SDR_K4(:,:,nn)] = waveform_design_SDR_covmat( d_theta,H_K4,M,4,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__ZF_K6(:,:,nn),p_K6(:,nn),W_ZF_K6(:,:,nn)] = waveform_design_ZF_covmat( d_theta,H_K6,M,6,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
        [R__SDR_K6(:,:,nn),r_K6,W_SDR_K6(:,:,nn)] = waveform_design_SDR_covmat( d_theta,H_K6,M,6,P,a,a_theta_bar,theta,sinr_threshold,noise_variance,Pt );
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
    % Beampattern MSE
    MSE_ZF_K2((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_ZF_K2*a)/real(trace(R_ZF_K2)),2).^2/length(d_theta);
    MSE_ZF_K4((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_ZF_K4*a)/real(trace(R_ZF_K4)),2).^2/length(d_theta);
    MSE_ZF_K6((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_ZF_K6*a)/real(trace(R_ZF_K6)),2).^2/length(d_theta);
    MSE_SDR_K2((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_SDR_K2*a)/real(trace(R_SDR_K2)),2).^2/length(d_theta);
    MSE_SDR_K4((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_SDR_K4*a)/real(trace(R_SDR_K4)),2).^2/length(d_theta);
    MSE_SDR_K6((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = norm(diag(a'*R0*a)/real(trace(R0)) - diag(a'*R_SDR_K6*a)/real(trace(R_SDR_K6)),2).^2/length(d_theta)


    INR_K2((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = INR_func(R_SDR_K2,H_K2,r_K2,M,2,noise_variance);
    INR_K4((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = INR_func(R_SDR_K4,H_K4,r_K4,M,4,noise_variance);
    INR_K6((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = INR_func(R_SDR_K6,H_K6,r_K6,M,6,noise_variance);
    SUM_RATE_SDR_K2((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = SUM_RATE_func(R_SDR_K2,H_K2,r_K2,M,2,noise_variance);
    SUM_RATE_SDR_K4((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = SUM_RATE_func(R_SDR_K4,H_K4,r_K4,M,4,noise_variance);
    SUM_RATE_SDR_K6((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = SUM_RATE_func(R_SDR_K6,H_K6,r_K6,M,6,noise_variance);
    SUM_RATE_ZF_K2((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = sum(log2(pK2/noise_variance));
    SUM_RATE_ZF_K4((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = sum(log2(pK4/noise_variance));
    SUM_RATE_ZF_K6((sinr_threshold-SINR_threshold_lower)/SINR_step+1) = sum(log2(pK6/noise_variance));
    
    clc;
    disp(['Progress SINR_threshold- ',num2str(sinr_threshold),'/',num2str(SINR_threshold_upper)]);
end

%% ----------------- Figures ----------------------
SINR_threshold = SINR_threshold_lower:SINR_step:SINR_threshold_upper;
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


figure(3); % Sum Rate versus SINR_threshold
plot(SINR_threshold,SUM_RATE_ZF_K2,'bx-','LineWidth',1.5,'MarkerSize',8);hold on;
plot(SINR_threshold,SUM_RATE_ZF_K4,'go-','LineWidth',1.5,'MarkerSize',8);hold on;
plot(SINR_threshold,SUM_RATE_ZF_K6,'r^-','LineWidth',1.5,'MarkerSize',8);hold on;
plot(SINR_threshold,SUM_RATE_SDR_K2,'b*-','LineWidth',1.5,'MarkerSize',8);hold on;
plot(SINR_threshold,SUM_RATE_SDR_K4,'g--','LineWidth',1.5,'MarkerSize',8);hold on;
plot(SINR_threshold,SUM_RATE_SDR_K6,'r+--','LineWidth',1.5,'MarkerSize',8);
grid on;
xlabel('SINR threshold (dB)');
ylabel('Sum Rate (bps/Hz)');
legend('ZF,K=2','ZF,K=4','ZF,K=6','SDR,K=2','SDR,K=4','SDR,K=6');
% figure(4); % Interference-noise-ratio versus SINR_threshold
% semilogy(SINR_threshold,INR_K2,'bx-','LineWidth',1.5,'MarkerSize',8);hold on;
% semilogy(SINR_threshold,INR_K4,'go-','LineWidth',1.5,'MarkerSize',8);hold on;
% semilogy(SINR_threshold,INR_K6,'r^-','LineWidth',1.5,'MarkerSize',8);hold on;
% grid on;
% xlabel('SINR threshold (dB)');
% ylabel('Interference-noise-ratio dB');
% legend('ZF,K=2','ZF,K=4','ZF,K=6','SDR,K=2','SDR,K=4','SDR,K=6');