%{
    Online supplementary materials of the paper titled:
    Distributionally Robust Adaptive Beamforming
    By Shixiong Wang, Wei Dai, and Geoffrey Ye Li
    From Department of Electrical and Electronic Engineering, Imperial College London

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     8 Oct 2024, 13 March 2025
    @Home:     https://github.com/Spratm-Asleaf/Beamforming-UDL
%}


%% Output SINR Against SNR

SNRLength = length(SNR);

SINR_Capon0     = zeros(MonteCarlo, SNRLength);
SINR_Capon      = zeros(MonteCarlo, SNRLength);
SINR_Capon_DL   = zeros(MonteCarlo, SNRLength);
SINR_IPN0       = zeros(MonteCarlo, SNRLength);
SINR_IPN        = zeros(MonteCarlo, SNRLength);
SINR_IPN_DL     = zeros(MonteCarlo, SNRLength);
SINR_IPN_UDL    = zeros(MonteCarlo, SNRLength);
SINR_IPN_MatEnt = zeros(MonteCarlo, SNRLength);

L = Snapshot;

%% Main Loop
for i = 1:SNRLength
    % SNR and noise power
    snr = 10^(SNR(i)/10);
    Pv = Ps(1)/snr;

    % True Cov of snapshot
    Rx0 = a0(N, Theta)*diag(Ps)*a0(N, Theta)' + Pv*eye(N);
    % True Cov of IPN
    Ripn0 = Rx0 - Ps(1)*a0(N, Theta(1))*a0(N, Theta(1))';

    for mc = 1:MonteCarlo
        %% Signal transmission abd reception
        % Radiation signals
        S = diag(sqrt(Ps)) * sqrt(0.5) * (randn(K, L) + 1j*randn(K, L));
        % Channel noise
        V = sqrt(Pv)       * sqrt(0.5) * (randn(N, L) + 1j*randn(N, L));
        % Propagation
        X = a0(N, Theta)*S + V;
        
        %% Sample covariance and its eigen-decomposition
        Rx = X*X'/L;
        [U, D] = eig(Rx);
        [d, index] = sort(diag(D), 'descend');
        D = diag(d);
        U = U(:, index);

        %% Steer Vectors
        % True Steering Vector
        stVec0 = a0(N, Theta(1));
        % Assumed Steering Vector
        % DoA Mismatch
        % a = @(N, Theta) a0(N, Theta + Delta*randn);
        % Calibration Error
        a = @(N, Theta) a0(N, Theta) + Delta*randn(N, 1);
        
        if isSteerVectorUncertain
            stVec = a(N, Theta(1));          %  Deviated from true steer vector stVec0
        else
            stVec = stVec0;                  %  Same as true steer vector stVec0
        end
        
        %% Calculate SINRs
        %% IPN-Estimation Based Beamformers
        % True-IPN-based beamformer with true steer vector and true data covariacne
        % Optimal Beamformer
        w_IPN0 = ((Ripn0^-1)*stVec0)/(stVec0'*(Ripn0^-1)*stVec0);
        SINR_IPN0(mc, i) = GetSINR(w_IPN0, stVec0, Ripn0, Ps(1));

        % Estimated-IPN-based beamformer with assumed steer vector and estimated data covariacne
        Ripn = GetEstimatedRipn(Rx, ThetaIPN, a);
        w_IPN = ((Ripn^-1)*stVec)/(stVec'*(Ripn^-1)*stVec);
        SINR_IPN(mc, i) = GetSINR(w_IPN, stVec0, Ripn0, Ps(1));
    
        % Diagonally-Loaded (DL) Estimated-IPN-based beamformer with assumed steer vector and estimated data covariacne
        epsilon = 0.01;
        w_IPN_DL = (((Ripn + epsilon * eye(N))^-1)*stVec)/(stVec'*((Ripn + epsilon * eye(N))^-1)*stVec);
        SINR_IPN_DL(mc, i) = GetSINR(w_IPN_DL, stVec0, Ripn0, Ps(1));

        % Unbalanced Diagonally-Loaded (UDL) Estimated-IPN-based beamformer with assumed steer vector and estimated data covariacne
        delta1 = 10;
        delta2 = 0.01;
        if Scenario == 1
            epsilon = 0;  % Different Scenario, Different Parameter
        end
        Rx_UDL = U*(D + blkdiag(delta1*eye(KK), delta2*eye(N-KK)))*U';
        Ripn = GetEstimatedRipn(Rx_UDL, ThetaIPN, a);
        w_IPN_UDL = (((Ripn + epsilon * eye(N))^-1)*stVec)/(stVec'*((Ripn + epsilon * eye(N))^-1)*stVec);
        SINR_IPN_UDL(mc, i) = GetSINR(w_IPN_UDL, stVec0, Ripn0, Ps(1));

        % Maximum-Entropy Estimated-IPN-based beamformer with assumed steer vector and estimated data covariacne
        u = [1; zeros(N-1, 1)];
        Rx_MatEnt = (Rx^-1 * (u * u') * Rx^-1 + 1e-5*eye(N))^-1;
        Ripn = GetEstimatedRipn(Rx_MatEnt, ThetaIPN, a);
        w_IPN_MatEnt = ((Ripn^-1)*stVec)/(stVec'*(Ripn^-1)*stVec);
        SINR_IPN_MatEnt(mc, i) = GetSINR(w_IPN_MatEnt, stVec0, Ripn0, Ps(1));

        %% Capon Beamformers
        % Capon with true steer vector and true data covariacne
        % Totally amounts to the "Optimal Beamformer" before
        w_Capon0 = ((Rx0^-1)*stVec0)/(stVec0'*(Rx0^-1)*stVec0);
        SINR_Capon0(mc, i) = GetSINR(w_Capon0, stVec0, Ripn0, Ps(1));
    
        % Capon with assumed steer vector and estimated data covariacne
        w_Capon  = ((Rx^-1)*stVec)/(stVec'*(Rx^-1)*stVec);
        SINR_Capon(mc, i)  = GetSINR(w_Capon, stVec0, Ripn0, Ps(1));
       
        % Diagonally-Loaded (DL) Capon with assumed steer vector and estimated data covariacne
        epsilon = 0.1;
        w_Capon_DL = (((Rx + epsilon * eye(N))^-1)*stVec)/(stVec'*((Rx + epsilon * eye(N))^-1)*stVec);
        SINR_Capon_DL(mc, i) = GetSINR(w_Capon_DL, stVec0, Ripn0, Ps(1));
    end
end


