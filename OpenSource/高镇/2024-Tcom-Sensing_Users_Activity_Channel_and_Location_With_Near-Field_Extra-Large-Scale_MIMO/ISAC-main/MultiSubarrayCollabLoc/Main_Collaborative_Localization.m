clc; clear;
close all;
% rng(1,'twister');
M_suba = 10;
N_sub = 8;
FacSp = 0.8;
N_BS = M_suba*N_sub;
N_RF = M_suba;
fs = 200e6;     % 200 MHz
fc = 28e9;      % 28 GHz
c_light = 3e8;
lambda = c_light/fc;
d_ant = lambda/2;
Delta_suba = 5;
N_subc = 2048;   % 512 1024 2048
delta_f = fs/N_subc;
tau_max = 100/3e8;
N_cp = tau_max*fs;
Delta_f = 1/tau_max;
delta_subc = ceil(Delta_f/delta_f);
P_subc = ceil(N_subc/delta_subc);
Subc_Index_set = 1:delta_subc:N_subc;
K = 60;
Ka = 6;
Lp_max = 5;
sigma_2_alpha = 1;  % variance of path gain
awgn_en = 1;        % 1: add noise; 0: no noise
Index_Sub_Array_BS = reshape(1:N_BS,N_sub,M_suba);
DFT_mtx_sub = kron(eye(M_suba),dftmtx(N_sub))/sqrt(N_sub);

%% Set snr
sigma2_NSD = -174;	% noise power spectral density（NPSD）dBm/Hz
sigma2_no = 1e-3*10^(sigma2_NSD/10)*fs; % unit: W
sigma_no = sqrt(sigma2_no);
epsilon = sigma2_no;
% G_ofdm_set >= Ka*N_sub
G_ofdm = 50; % 50 55 60 65;
Ptx_set = 10:5:30;

Num_sim = 10000;
%
Pe_GMMV_BSOMP_set = zeros(length(Ptx_set),Num_sim);
NMSE_GMMV_BSOMP_set = zeros(length(Ptx_set),Num_sim);
MSE_Loca = zeros(length(Ptx_set),Num_sim);
%% Start
tic
% I0=0;  % debug
% A0_startmatlabpool(4)
for nn_s = 1:Num_sim
    % Generating Near Field Channel
    AUD_Index_K_set = sort(randperm(K,Ka));
    [H_set, Loca_Censub, Loca_UEs, D] = New_Localiz_Channel_Model(FacSp, AUD_Index_K_set, M_suba, N_sub, fs, fc, Delta_suba, Subc_Index_set, N_subc, K, Ka, Lp_max, sigma_2_alpha);
    Active_flag = zeros(K,1);
    Active_flag(AUD_Index_K_set) = 1;
    
    Y_GMMV_ALL_set = zeros(G_ofdm*M_suba,P_subc);
    Y_GMMV_P2 = zeros(G_ofdm*M_suba,P_subc);
    Z_GMMV_ALL_set = zeros(G_ofdm*M_suba,K*N_BS,P_subc);
    for gg_1 = 1:G_ofdm
        % Design combining matrix
        W_RF_g = zeros(N_BS,M_suba);
        for mm_1 = 1:M_suba
            W_RF_g(Index_Sub_Array_BS(:,mm_1),mm_1) = exp(1i*2*pi*rand(N_sub,1))/sqrt(N_sub); %
        end
        W_g_set = zeros(N_BS,M_suba,P_subc);
        for pp_1 = 1:P_subc
            W_BB_g_p = eye(M_suba);
            W_g_set(:,:,pp_1) = W_RF_g*W_BB_g_p;
        end
        s_pilot_g_set = exp(-1i*2*pi*rand(K,P_subc));
        Noise_set_g = sigma_no*(normrnd(0,1,N_BS,P_subc) + 1i*normrnd(0,1,N_BS,P_subc))/sqrt(2);

        Rx_Sig_Gmmv =pagemtimes(permute(H_set(:,:,AUD_Index_K_set),[1,3,2]),reshape(s_pilot_g_set(AUD_Index_K_set,:),[Ka,1,P_subc]));
        Y_GMMV_ALL_set((gg_1-1)*M_suba+1:gg_1*M_suba,:) = reshape(pagemtimes(permute(conj(W_g_set),[2,1,3]),(Rx_Sig_Gmmv)), [M_suba, P_subc]);

        Y_GMMV_P2((gg_1-1)*M_suba+1:gg_1*M_suba,:) = reshape(pagemtimes(permute(conj(W_g_set),[2,1,3]),(reshape(awgn_en*Noise_set_g,[M_suba*N_sub, 1,P_subc]))), [M_suba, P_subc]);
        for pp_2 = 1:P_subc
            Z_GMMV_ALL_set((gg_1-1)*M_suba+1:gg_1*M_suba,:,pp_2) = kron(s_pilot_g_set(:,pp_2).',W_g_set(:,:,pp_2)');
        end
    end
    
    %%
    for ptx = 1:length(Ptx_set) % parfor or for
	% parfor ptx = 1:length(Ptx_set) % parfor or for
        Ptx_dBm = Ptx_set(ptx);   % Transmit power
        Ptx = 10^(Ptx_dBm/10)*1e-3;
        %% Signal Transmission Process
        Y_GMMV_set = sqrt(Ptx) * Y_GMMV_ALL_set;
        Z_GMMV_set = sqrt(Ptx) * Z_GMMV_ALL_set;

        Y_GMMV_set = Y_GMMV_set + Y_GMMV_P2;
        
        %% Active User Detection and Channel Estimation
        %% GMMV_BSOMP_AUD_CE Algorithm with considering angle-domain
        % [AUD_Index_set_GMMV_BSOMP, H_set_GMMV_BSOMP] = A32_GMMV_BSOMP_AUD_CE(Y_GMMV_set, Z_GMMV_set, K, Ka, DFT_mtx_sub);
        % [AUD_Index_set_GMMV_BSOMP, H_set_GMMV_BSOMP] = GMMV_OMP_subA(Y_GMMV_set, Z_GMMV_set, K, epsilon, M_suba);
        % tic
        [AUD_Index_set_GMMV_BSOMP, H_set_GMMV_BSOMP] = GMMV_OMP_subA_adv2_MT(Y_GMMV_set, Z_GMMV_set, K, epsilon, M_suba);
        % [AUD_Index_set_GMMV_BSOMP, H_set_GMMV_BSOMP] = GMMV_OMP_subA_adv2_MT(Y_GMMV_set, Z_GMMV_set, K, epsilon, M_suba);
        Active_flag_GMMV_BSOMP = zeros(K,1);
        Active_flag_GMMV_BSOMP(AUD_Index_set_GMMV_BSOMP) = 1;
        % AUD Error Probability (Pe)
        Pe_GMMV_BSOMP_gg_nn = sum(abs(Active_flag - Active_flag_GMMV_BSOMP))/K;
        Pe_GMMV_BSOMP_set(ptx,nn_s) = Pe_GMMV_BSOMP_gg_nn;
        % NMSE
        NMSE_GMMV_BSOMP_gg_nn = 10*log10(norm(H_set(1:end)-H_set_GMMV_BSOMP(1:end))^2/norm(H_set(1:end))^2);
        NMSE_GMMV_BSOMP_set(ptx,nn_s) = NMSE_GMMV_BSOMP_gg_nn;
        % toc

        ACTT=[];
        for iii=1:Ka
            if ~isempty(find(AUD_Index_set_GMMV_BSOMP==AUD_Index_K_set(iii),1))
                ACTT=[ACTT,AUD_Index_K_set(iii)];
            end
        end
        if isempty(ACTT)
            MSE_Loca(ptx,nn_s) =1;
            continue;
        end
        Loca_UEs_act=Loca_UEs(ACTT,:);
        He_act = H_set_GMMV_BSOMP(:, :, ACTT);
        [xyAxis,MSE] = FuncLocation(He_act, N_sub, M_suba, Loca_Censub, Loca_UEs_act, fs,Subc_Index_set,N_subc);

        MSE_Loca(ptx,nn_s) =MSE;
        
        
        %% Display
        if mod(nn_s, 10) == 0
            Pe_GMMV_BSOMP = sum(Pe_GMMV_BSOMP_set,2)/nn_s;
            NMSE_GMMV_BSOMP_dB = 10*log10(sum(NMSE_GMMV_BSOMP_set,2)/nn_s);
            MSE_Loca_dB = sum(MSE_Loca,2)/nn_s;
            save Localiza_Ptx_fs200.mat fs FacSp N_sub nn_s Pe_GMMV_BSOMP_set NMSE_GMMV_BSOMP_set Ptx_set ...
                G_ofdm Pe_GMMV_BSOMP NMSE_GMMV_BSOMP_dB MSE_Loca_dB MSE_Loca
        end
    end
    disp(['  G_ofdm = ' num2str(G_ofdm) ' , num_sim = ' num2str(nn_s) ' , Ptx = ' num2str(Ptx_dBm) ' dBm'])

    toc
end

Pe_GMMV_BSOMP = sum(Pe_GMMV_BSOMP_set,2)/nn_s;
NMSE_GMMV_BSOMP_dB = 10*log10(sum(NMSE_GMMV_BSOMP_set,2)/nn_s);
MSE_Loca_dB = sum(MSE_Loca,2)/nn_s;
save Localiza_Ptx_fs200.mat fs FacSp N_sub nn_s Pe_GMMV_BSOMP_set NMSE_GMMV_BSOMP_set Ptx_set ...
    G_ofdm Pe_GMMV_BSOMP NMSE_GMMV_BSOMP_dB MSE_Loca_dB MSE_Loca


% A0_closematlabpool
toc
disp('Finished all');
%
%% Plot
MarkerSize = 6;
LineWidth = 2;
Fontsize = 12;

figure;
hold on;
legend on;
box on;
for i = 1:size(MSE_Loca,1)
    cdfplot(sqrt(MSE_Loca(i,:)))
end
set(gca, 'XScale', 'log');