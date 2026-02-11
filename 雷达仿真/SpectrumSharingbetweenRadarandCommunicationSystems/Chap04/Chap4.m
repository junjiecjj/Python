clear;
clc;
close all;

% 4.10.1 Overlapped-MIMO Main Module

% Total number of transmitting antennas
M = 20;
% Total number of receiving antennas
M_r = 20;
% transmitter spacings in wavelength
d_t = 0.5;
% Direction of target
theta_tar = 15*pi/180;
% Uplink steering vector
a_tar = exp(-j*d_t*2*pi*(0:M-1)'*sin(theta_tar));
% Downlink steering vector
b_tar = exp(-j*pi*(0:M_r-1)'*sin(theta_tar));
% K is the number of subarrays in the MIMO
no_subarrays = [1 5 10 20];
% Overall beampattern
Rx_pattern_conv = [];
% Overall beampattern w/ projection
Rx_pattern_conv_proj = [];

for ksub = 1:length(no_subarrays);
    % Number of antennas in each subarray
    K_sub = no_subarrays(ksub);
    M_sub = M - K_sub + 1;
    W_u_conv = uplinkbeamform(a_tar, K_sub, M_sub);
    % Computing Transmit and diversity beampatterns
    Theta_grid = [linspace(-pi/2,pi/2,180)];
    % Compute virtual steering vectors of target
    [v_tar]      = virtual_sv_proj(theta_tar, M, d_t, M_r, M_sub, K_sub, W_u_conv);
    [v_tar_proj] = virtual_sv_proj(theta_tar, M, d_t, M_r, M_sub, K_sub, W_u_conv);

    % Conventional downlink beamformer
    w_d_conv = v_tar/(norm(v_tar));
    % Conventional downlink beamformer
    w_d_conv_proj = v_tar_proj/(norm(v_tar_proj));

    % Compute and plot overall Tx/Rx beampattern
    % Use this to plot overall beampattern
    w_d = w_d_conv;
    % Use this to plot overall beampattern
    w_d_proj = w_d_conv_proj;
    size_w_d = size(w_d);
    [V_grid]      = virtual_sv_proj(Theta_grid, M, d_t, M_r, M_sub, K_sub, W_u_conv);
    [V_grid_proj] = virtual_sv_proj(Theta_grid, M, d_t, M_r, M_sub, K_sub, W_u_conv);
    Rx_pattern      = [10*log10(abs(w_d'*V_grid).^2)];
    Rx_pattern_proj = [10*log10(abs(w_d'*V_grid_proj).^2)];
    Rx_pattern = Rx_pattern - max(Rx_pattern);
    Rx_pattern_conv = [Rx_pattern_conv; Rx_pattern];
    Rx_pattern_proj = Rx_pattern_proj - max(Rx_pattern_proj);
    Rx_pattern_conv_proj = [Rx_pattern_conv_proj; Rx_pattern_proj];
end

% --- Plots --- 
% Fig. 4.3
figure(1);
Theta = Theta_grid;
plot(Theta*180/pi,1.02*Rx_pattern_conv(1,:),'g', ...
     Theta*180/pi,1.0*Rx_pattern_conv(2,:),'r', ...
     Theta*180/pi,Rx_pattern_conv(3,:),'b', ...
     Theta*180/pi,Rx_pattern_conv(4,:),'k--','linewidth',2)
grid
axis([-90 90 -120 30])
xlabel('Angle (deg)')
ylabel('Overall Gain (dB)')
legend('Overlapped-MIMO Radar (K=1)', ...
       'Overlapped-MIMO Radar (K=5)', ...
       'Overlapped-MIMO Radar (K=10)', ...
       'MIMO Radar (K=20)')
% Fig. 4.4
figure(2);
Theta = Theta_grid;
plot(Theta*180/pi,1.02*Rx_pattern_conv_proj(1,:),'g', ...
     Theta*180/pi,1.0*Rx_pattern_conv_proj(2,:),'r', ...
     Theta*180/pi,Rx_pattern_conv_proj(3,:),'b', ...
     Theta*180/pi,Rx_pattern_conv_proj(4,:),'k--','linewidth',2)
axis([-90 90 -120 30])
xlabel('Angle (deg)')
ylabel('Overall Gain (dB)')
legend('Overlapped-MIMO Radar w/ NSP (K=1)', ...
       'Overlapped-MIMO Radar w/ NSP (K=5)', ...
       'Overlapped-MIMO Radar w/ NSP (K=10)', ...
       'Overlapped-MIMO Radar w/ NSP (K=20)')

% 4.10.2 Uplink Beamforming Matrix
% This function computes the the uplink beamforming matrix.
% The kth column respresents the beamforming weight vector
% of the kth subarray
function [W_u] = uplinkbeamform(a_tar, K_sub, M_sub)
    % Uplink weight vector (equal for all subarrays)
    w_u = a_tar(1:M_sub);
    size_w_u = size(w_u);
    % Uplink weight vector normalized
    w_u = w_u/(norm(w_u));
    % Weights that are equal for all subarrays
    W_u = kron(ones(1, K_sub), w_u);
end

% 4.10.3 Virtual Steering Vector 
% This function computes the virtual steering vectors
% with projection
function [v_sv] = virtual_sv_proj(Theta, Mt, d_t, Mr, M_sub, no_subarrays, W_u);
    % No of receive Antennas
    N_R = 15;
    
    % ---TX and RX Array ---%
    Tx_sv = exp(-j*d_t*2*pi * (0:Mt-1)'*sin(Theta));
    % Receiving steering vectors
    Rx_sv = exp(-j*pi * (0:Mr-1)'*sin(Theta));
    
    % --- VIRTUAL ARRAY W/O A_R ---%
    P = [];
    H = randn(N_R, no_subarrays * M_sub) + j * randn(N_R, no_subarrays * M_sub);
    P = null(H) * ctranspose(null(H));
    v_sv2 = [];
    for kk = 1:no_subarrays
        for mm = 1:M_sub
            v_temp2 = [];
            w_u = W_u(mm,kk);
            for jj = 1:length(Theta)
                v_temp2 = [v_temp2, (w_u' * Tx_sv(kk + mm -1, jj)) ];
            end
            v_sv2 = [v_sv2; v_temp2];
        end
    end
    v_sv2_P = P * v_sv2;
    
    v_sv3 = [];
    for vv = 1:no_subarrays * M_sub
        v_temp3 = [];
        for jj = 1:length(Theta)
            v_temp3 = [v_temp3, (v_sv2_P(vv, jj)) * Rx_sv(:, jj)];
        end
        v_sv3 = [v_sv3; v_temp3];
    end
     v_sv = v_sv3;
end



% 4.10.4 Number of Subarray 
% Determine the number of subarray
K1 = 1:1:10;
K2 = 1:1:15;
K3 = 1:1:20;
M1 = 10;
M2 = 15;
M3 = 20;

for ii = 1:1:10
    ME1(ii) = (M1-ii+1)*ii;
end

for jj = 1:1:15
    ME2(jj) = (M2-jj+1)*jj;
end

for kk = 1:1:20
    ME3(kk) = (M3-kk+1)*kk;
end
% Fig. 4.5
figure(3);
plot(K1,ME1,'g-d', 'linewidth',2), grid
hold on
plot(K2,ME2,'b-s','linewidth',2)
plot(K3,ME3,'r-o','linewidth',2)
xlabel('K')
ylabel('M_{epsilon}')
legend('M_T = 10','M_T = 15','M_T = 20')
axis([0 21 0 120])