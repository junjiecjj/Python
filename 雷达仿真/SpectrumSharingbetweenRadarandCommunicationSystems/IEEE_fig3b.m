% Compares Pd for Multi BS side-to-side with orthogonal waveforms 
% Select Mt number 
% Select BS number
clear;
clc;
close all;
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
SNR_db = -8:1:10; 
SNR_mag = 10.^(SNR_db./10); 
%Probability of false alarm values 
P_FA = [10^-5];
% P_FA = [10^-1, 10^-2];
%% Monte-Carlo iterations 
MC_iter = 20; 
Pd_orthog_cell = cell(1,MC_iter); 
Pd_NSP_cell = cell(1,MC_iter); 
for i=1:MC_iter %% Interference channel matrix H generation and null space computation
    % Generate Cellular Channels and Find NS of them and select 
    % Channel with Min NS 
    BS = 5; 
    % Make a cell to store matrices 
    BS_channels = cell(1,BS); 
    % Make a cell to store Projectors for every BS
    Proj_matrix = cell(1,BS);
    for b = 1:BS
        BS_channels{b} = (randn(Nr,Mt)+1i*randn(Nr,Mt)); 
        Proj_matrix{b} = null(BS_channels{b}) * ctranspose(null(BS_channels{b})); 
        Rs_null{b} = Proj_matrix{b} * Rs * Proj_matrix{b}';
        % Non-centrality parameter of chi-square
        for z = 1:length(SNR_mag)
            rho_orthog(b) = SNR_mag(z)*(abs(a'*Rs.'*a))^2;
            rho_NSP(b) = SNR_mag(z)*(abs(a'*Rs_null{b}.'*a))^2;
            % Creates threshold values for a desired Pfa for an inverse central-chi-square w/2 degrees of freedom
            delta = chi2inv(ones(1, length(P_FA)) - P_FA, repmat(2,1,length(P_FA)));
            % rows = SNR, cols = P_FA, ncx2cdf = Noncentral chi -square cumulative distribution function
            Pd_orthog(z,:) = ones(1, length(P_FA)) - ncx2cdf(delta, repmat(2, 1, length(P_FA)), repmat(rho_orthog(b), 1, length(P_FA )));
            Pd_NSP(z,:) = ones(1, length(P_FA)) - ncx2cdf(delta,repmat(2, 1, length(P_FA)), repmat(rho_NSP(b), 1, length(P_FA)));
        end
        Pd_orthog_cell{b}= Pd_orthog;
        Pd_NSP_cell{b}= Pd_NSP;
    end 
    Pd_orthog_cat(:,:,i) = cell2mat(Pd_orthog_cell);
    Pd_NSP_cat(:,:,i) = cell2mat(Pd_NSP_cell);
end
Pd_orthog_cat_mean = mean(Pd_orthog_cat ,3);
Pd_NSP_cat_mean = mean(Pd_NSP_cat ,3);
%% Plots Probability of detection curves for given  Probability of false alarm 
figure
plot(SNR_db',Pd_NSP_cat_mean(:,1),'g','LineWidth',2.5); hold on;
plot(SNR_db',Pd_NSP_cat_mean(:,2),'b','LineWidth',2.5);
plot(SNR_db',Pd_NSP_cat_mean(:,3),'r','LineWidth',2.5);
plot(SNR_db',Pd_NSP_cat_mean(:,4),'m','LineWidth',2.5);
plot(SNR_db',Pd_NSP_cat_mean(:,5),'y','LineWidth',2.5);
plot(SNR_db',Pd_orthog_cat_mean(:,1),'k','LineWidth',2.5)
xlabel('SNR','fontsize' ,14);
ylabel('P_D','fontsize' ,14);
title('P_D for P_{FA} = 10^{-5}','fontsize' ,14);
legend('P_D for NSP Waveforms to BS 1', 'P_D for NSP Waveforms to BS 2', 'P_D for NSP Waveforms to BS 3', 'P_D for NSP Waveforms to BS 4', 'P_D for NSP Waveforms to BS 5', 'P_D for Orthogonal Waveforms')





