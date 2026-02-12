% Compares Pd for Multi BS side-to-side with orthogonal waveforms 
% Select Mt number 
% Select BS number
clear;
clc;
close all;

rng(42,'twister');
%% Define Parameters 
% Speed of light 
c = 3*10^8; 
% Nr Comm Receivers 
Nr = 2;
% Mt Radar Transmiters 
Mt = 8; 
% Mr Radar Receivers
Mr = Mt; 
% Radial velocity of 2000 m/s 
v_r = 2000; 
% Radar reference point 
r_0 = 500*10^3; 
% Carrier frequency 3.5GHz 
f_c = 3.5*10^9; % Angular carrier frequency 
omega_c = 2*pi*f_c; 
lambda = (2*pi*c)/omega_c; 
theta = 10;
%% Steering vector and Transmit Signal Correlation Matrix 
% Transmit/Receive Steering vector (Mt x 1)
a = [1 exp(1i * pi *(1:Mt-1)* sin(theta))]'; 
% Transmit Correlation Matrix (Mt x Mt) for Orthonormal Waveforms
Rs = eye(Mt);
%% Define SNR for ROC (Reciever Operating Characteristics)
SNR_db = -10:1:30; 
SNR_mag = 10.^(SNR_db./10); 
%Probability of false alarm values 
P_FA = [10^-1, 10^-3, 10^-5, 10^-7];
% P_FA = [10^-1, 10^-5];
%% Monte-Carlo iterations 
MC_iter = 100; 
BS = 5;

Pd_orthog_it = zeros(MC_iter, BS, length(SNR_mag), length(P_FA));
Pd_NSP_it = zeros(MC_iter, BS, length(SNR_mag), length(P_FA));
for i=1:MC_iter
    Pd_orthog_bs = zeros(BS, length(SNR_mag), length(P_FA));
    Pd_NSP_bs = zeros(BS, length(SNR_mag), length(P_FA));
    for b = 1:BS
        BS_channels = (randn(Nr,Mt)+1i*randn(Nr,Mt)); 
        Proj_matrix = null(BS_channels) * ctranspose(null(BS_channels)); 
        Rs_null     = Proj_matrix * Rs * Proj_matrix';
        Pd_orthog = zeros(length(SNR_mag), length(P_FA));
        Pd_NSP    = zeros(length(SNR_mag), length(P_FA));
        % Non-centrality parameter of chi-square
        for z = 1:length(SNR_mag)
            rho_orthog = SNR_mag(z)*(abs(a'*Rs.'*a))^2;
            rho_NSP    = SNR_mag(z)*(abs(a'*Rs_null.'*a))^2;
            % Creates threshold values for a desired Pfa for an inverse central-chi-square w/2 degrees of freedom
            delta = chi2inv(ones(1, length(P_FA)) - P_FA, repmat(2, 1, length(P_FA)));
            % rows = SNR, cols = P_FA, ncx2cdf = Noncentral chi -square cumulative distribution function
            Pd_orthog(z,:) = ones(1, length(P_FA)) - ncx2cdf(delta, repmat(2, 1, length(P_FA)), repmat(rho_orthog, 1, length(P_FA )));
            Pd_NSP(z,:) = ones(1, length(P_FA)) - ncx2cdf(delta,repmat(2, 1, length(P_FA)), repmat(rho_NSP, 1, length(P_FA)));
        end
        Pd_orthog_bs(b,:,:) = Pd_orthog;
        Pd_NSP_bs(b,:,:) = Pd_NSP;
    end
    Pd_orthog_it(i,:,:,:) = Pd_orthog_bs;
    Pd_NSP_it(i,:,:,:) = Pd_NSP_bs;
end

Pd_orthog_mean = mean(Pd_orthog_it ,1);
Pd_NSP_mean = mean(Pd_NSP_it ,1);


%% Plots Probability of detection curves for given  Probability of false alarm
colors = ['g', 'b', 'r', 'm', 'y' ];
for z = 1:length(P_FA)
    figure(z);
    for b = 1:1
        plot(SNR_db',squeeze(Pd_NSP_mean(1, b,:,z)),colors(b),'LineWidth',2.5); hold on;
    end
    plot(SNR_db',squeeze(Pd_orthog_mean(1, 1,:,z)),'k','LineWidth',2.5); hold on;
    xlabel('SNR','fontsize' ,14);
    ylabel('P_D','fontsize' ,14);
    str = sprintf('P_D for P_{FA} = %.1e',P_FA(z)); 
    title(str,'fontsize' ,14);
    legend('P_D for NSP Waveforms', 'P_D for Orthogonal Waveforms')
end




