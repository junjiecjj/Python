clc; clear;

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
Ptx_set = 0:10:30;

Num_sim = 1e5;
%
Pe_BSAMP_set = zeros(length(Ptx_set),Num_sim);
NMSE_BSAMP_set = zeros(length(Ptx_set),Num_sim);
%
Pe_BSP_set = zeros(length(Ptx_set),Num_sim);
NMSE_BSP_set = zeros(length(Ptx_set),Num_sim);
%
Subc_index_temp = randperm(P_subc);
P_subc_BOMP = sort(Subc_index_temp(1:ceil(P_subc/3)));
Pe_BOMP_set = zeros(length(Ptx_set),Num_sim);
NMSE_BOMP_set = zeros(length(Ptx_set),Num_sim);
%
Pe_MMV_BSOMP_Ad_set = zeros(length(Ptx_set),Num_sim);
NMSE_MMV_BSOMP_Ad_set = zeros(length(Ptx_set),Num_sim);
%
Pe_MMV_BSOMP_set = zeros(length(Ptx_set),Num_sim);
NMSE_MMV_BSOMP_set = zeros(length(Ptx_set),Num_sim);
%
Pe_GMMV_BSOMP_Ad_set = zeros(length(Ptx_set),Num_sim);
NMSE_GMMV_BSOMP_Ad_set = zeros(length(Ptx_set),Num_sim);
%
Pe_GMMV_BSOMP_set = zeros(length(Ptx_set),Num_sim);
NMSE_GMMV_BSOMP_set = zeros(length(Ptx_set),Num_sim);
%
NMSE_Oracle_LS_set = zeros(length(Ptx_set),Num_sim);

%
Pe_OMPMMVsubA_set = zeros(length(Ptx_set),Num_sim);
NMSE_OMPMMVsubA_set = zeros(length(Ptx_set),Num_sim);
%
Pe_OMPMMVsubAdv_set = zeros(length(Ptx_set),Num_sim);
NMSE_OMPMMVsubAdv_set = zeros(length(Ptx_set),Num_sim);

%% Start
% A0_startmatlabpool(6)
tic
for nn_s = 1:Num_sim
    % Generating Near Field Channel
    AUD_Index_K_set = sort(randperm(K,Ka));
    H_set = New_Channel_Model(FacSp, AUD_Index_K_set, M_suba, N_sub, fs, fc, Delta_suba, Subc_Index_set, N_subc, K, Ka, Lp_max, sigma_2_alpha);
    Active_flag = zeros(K,1);
    Active_flag(AUD_Index_K_set) = 1;

    % Y_MMV_set = zeros(G_ofdm*M_suba,P_subc);
    % Z_MMV_set = zeros(G_ofdm*M_suba,K*N_BS);
    % Y_GMMV_set = zeros(G_ofdm*M_suba,P_subc);
    % Z_GMMV_set = zeros(G_ofdm*M_suba,K*N_BS,P_subc);
    % Y_GMMV_ALL_set = zeros(G_ofdm*M_suba,P_subc,length(Ptx_set));
    % Z_GMMV_ALL_set = zeros(G_ofdm*M_suba,K*N_BS,P_subc,length(Ptx_set));
    % W_all_set = zeros(N_BS,M_suba,P_subc,G_ofdm);
    Y_ALL_set = zeros(G_ofdm*M_suba,P_subc);
    Z_ALL_set = zeros(G_ofdm*M_suba,K*N_BS);
    Y_MMV_P2 = zeros(G_ofdm*M_suba,P_subc);
    % tic
    for gg_1 = 1:G_ofdm
        % Design combining matrix
        W_RF_g = zeros(N_BS,M_suba);
        for mm_1 = 1:M_suba
            W_RF_g(Index_Sub_Array_BS(:,mm_1),mm_1) = exp(1i*2*pi*rand(N_sub,1))/sqrt(N_sub); %
        end
        W_g_set = zeros(N_BS,M_suba,P_subc);
        for pp_1 = 1:P_subc
            %                 W_BB_g_p = exp(1i*2*pi*rand(M_suba,M_suba));
            W_BB_g_p = eye(M_suba);
            W_g_set(:,:,pp_1) = W_RF_g*W_BB_g_p;
        end
        % W_all_set(:,:,:,gg_1) = W_g_set;
        s_pilot_g_set = exp(-1i*2*pi*rand(K,P_subc));
        s_pilot_2 = repmat(s_pilot_g_set(:,1), 1, P_subc);
        % Noise
        Noise_set_g = sigma_no*(normrnd(0,1,N_BS,P_subc) + 1i*normrnd(0,1,N_BS,P_subc))/sqrt(2);

        % for iii = 1:length(Ptx_set)
        %     Ptx_dBm = Ptx_set(iii);   % Transmit power
        %     Ptx = 10^(Ptx_dBm/10)*1e-3;
        Rx_Sig_mmv =reshape(pagemtimes(permute(H_set(:,1:P_subc,AUD_Index_K_set(1:Ka)),[1,3,2]),reshape(s_pilot_2(AUD_Index_K_set(1:Ka),1:P_subc),[Ka,1,P_subc])), M_suba*N_sub, P_subc);
        Y_ALL_set((gg_1-1)*M_suba+1:gg_1*M_suba,:) = W_g_set(:,:,1)'*Rx_Sig_mmv;
        Z_ALL_set((gg_1-1)*M_suba+1:gg_1*M_suba,:) = kron(s_pilot_2(:,1).',W_g_set(:,:,1)');
        Y_MMV_P2((gg_1-1)*M_suba+1:gg_1*M_suba,:) = W_g_set(:,:,1)'*(awgn_en*Noise_set_g);
            % Rx_Sig_Gmmv =pagemtimes(permute(H_set(:,:,AUD_Index_K_set)*sqrt(Ptx),[1,3,2]),reshape(s_pilot_g_set(AUD_Index_K_set,:),[Ka,1,P_subc]));
            % Y_GMMV_ALL_set((gg_1-1)*M_suba+1:gg_1*M_suba,:,iii) = reshape(pagemtimes(permute(conj(W_g_set),[2,1,3]),(Rx_Sig_Gmmv)), [M_suba, P_subc]);
            % for pp_2 = 1:P_subc
            %     Z_GMMV_ALL_set((gg_1-1)*M_suba+1:gg_1*M_suba,:,pp_2,iii) = kron(sqrt(Ptx)*s_pilot_g_set(:,pp_2).',W_g_set(:,:,pp_2)');
            % end
        % end
    end
    % toc
    

    %%
    parfor ptx = 1:length(Ptx_set) % parfor or for
        Ptx_dBm = Ptx_set(ptx);   % Transmit power
        Ptx = 10^(Ptx_dBm/10)*1e-3;

        Y_MMV_set=sqrt(Ptx)*Y_ALL_set;
        Z_MMV_set=sqrt(Ptx)*Z_ALL_set;
        Y_MMV_set = Y_MMV_set + Y_MMV_P2;
        % Y_GMMV_set=Y_GMMV_ALL_set(:,:,ptx);
        % Z_GMMV_set=Z_GMMV_ALL_set(:,:,:,ptx);

        % for gg_1 = 1:G_ofdm
        %     Noise_set_g = sigma_no*(normrnd(0,1,N_BS,P_subc) + 1i*normrnd(0,1,N_BS,P_subc))/sqrt(2);
        %     W_g_set = W_all_set(:,:,:,gg_1);
        %     Y_MMV_set((gg_1-1)*M_suba+1:gg_1*M_suba,:) = Y_MMV_set((gg_1-1)*M_suba+1:gg_1*M_suba,:) + W_g_set(:,:,1)'*(awgn_en*Noise_set_g);
        %     % Y_GMMV_set((gg_1-1)*M_suba+1:gg_1*M_suba,:) = Y_GMMV_set((gg_1-1)*M_suba+1:gg_1*M_suba,:) +...
        %     %     reshape(pagemtimes(permute(conj(W_g_set),[2,1,3]),(reshape(awgn_en*Noise_set_g,[M_suba*N_sub, 1,P_subc]))), [M_suba, P_subc]);
        % end

        %% Active User Detection and Channel Estimation
        %% BSAMP_AUD_CE Algorithm
        [AUD_Index_set_BSAMP, H_set_BSAMP] = A23_BSAMP_AUD_CE(Y_MMV_set, Z_MMV_set, K, Ka);
        % [AUD_Index_set_BSAMP, H_set_BSAMP] = Copy_of_A22_BSP_AUD_CE(Y_MMV_set, Z_MMV_set, K, Ka);
        Active_flag_BSAMP = zeros(K,1);
        Active_flag_BSAMP(AUD_Index_set_BSAMP) = 1;
        % AUD Error Probability (Pe)
        Pe_BSAMP_gg_nn = sum(abs(Active_flag - Active_flag_BSAMP))/K;
        Pe_BSAMP_set(ptx,nn_s) = Pe_BSAMP_gg_nn;
        % NMSE
        NMSE_BSAMP_gg_nn = norm(H_set(1:end)-H_set_BSAMP(1:end))^2/norm(H_set(1:end))^2;
        NMSE_BSAMP_set(ptx,nn_s) = NMSE_BSAMP_gg_nn;

        %% BSP_AUD_CE Algorithm
        % [AUD_Index_set_BSP, H_set_BSP] = Copy_of_A23_BSAMP_AUD_CE(Y_MMV_set, Z_MMV_set, K, Ka);
        [AUD_Index_set_BSP, H_set_BSP] = A22_BSP_AUD_CE(Y_MMV_set, Z_MMV_set, K, Ka);
        Active_flag_BSP = zeros(K,1);
        Active_flag_BSP(AUD_Index_set_BSP) = 1;
        % AUD Error Probability (Pe)
        Pe_BSP_gg_nn = sum(abs(Active_flag - Active_flag_BSP))/K;
        Pe_BSP_set(ptx,nn_s) = Pe_BSP_gg_nn;
        % NMSE
        NMSE_BSP_gg_nn = norm(H_set(1:end)-H_set_BSP(1:end))^2/norm(H_set(1:end))^2;
        NMSE_BSP_set(ptx,nn_s) = NMSE_BSP_gg_nn;

        %% BOMP_AUD_CE Algorithm
        [AUD_Index_set_BOMP, H_set_BOMP] = MMV_OMP_A(Y_MMV_set, Z_MMV_set, K, epsilon, M_suba);
        % [AUD_Index_set_BOMP, H_set_BOMP] = A21_BOMP_AUD_CE(Y_MMV_set, Z_MMV_set, K, P_subc_BOMP, epsilon);
        Active_flag_BOMP = zeros(K,1);
        Active_flag_BOMP(AUD_Index_set_BOMP) = 1;
        % AUD Error Probability (Pe)
        Pe_BOMP_gg_nn = sum(abs(Active_flag - Active_flag_BOMP))/K;
        Pe_BOMP_set(ptx,nn_s) = Pe_BOMP_gg_nn;
        % NMSE
        % H_BOMP_set = H_set(:,P_subc_BOMP,:);
        NMSE_BOMP_gg_nn = norm(H_set(1:end)-H_set_BOMP(1:end))^2/norm(H_set(1:end))^2;
        NMSE_BOMP_set(ptx,nn_s) = NMSE_BOMP_gg_nn;

        %% OMP MMV Sub-Array Algorithm
        [AUD_Index_set_OMPMMVsubA, H_set_OMPMMVsubA] = MMV_OMP_subA(Y_MMV_set, Z_MMV_set, K, epsilon, M_suba);
        Active_flag_OMPMMVsubA = zeros(K,1);
        Active_flag_OMPMMVsubA(AUD_Index_set_OMPMMVsubA) = 1;
        % AUD Error Probability (Pe)
        Pe_OMPMMVsubA_gg_nn = sum(abs(Active_flag - Active_flag_OMPMMVsubA))/K;
        Pe_OMPMMVsubA_set(ptx,nn_s) = Pe_OMPMMVsubA_gg_nn;
        % NMSE
        NMSE_OMPMMVsubA_gg_nn = norm(H_set(1:end)-H_set_OMPMMVsubA(1:end))^2/norm(H_set(1:end))^2;
        NMSE_OMPMMVsubA_set(ptx,nn_s) = NMSE_OMPMMVsubA_gg_nn;

        %% OMP MMV Sub-Array Adv Algorithm
        [AUD_Index_set_OMPMMVsubAdv, H_set_OMPMMVsubAdv, Z_ind] = MMV_StrBOMP(Y_MMV_set, Z_MMV_set, K, epsilon, M_suba);
        Active_flag_OMPMMVsubAdv = zeros(K,1);
        Active_flag_OMPMMVsubAdv(AUD_Index_set_OMPMMVsubAdv) = 1;
        % AUD Error Probability (Pe)
        Pe_OMPMMVsubAdv_gg_nn = sum(abs(Active_flag - Active_flag_OMPMMVsubAdv))/K;
        Pe_OMPMMVsubAdv_set(ptx,nn_s) = Pe_OMPMMVsubAdv_gg_nn;
        % NMSE
        NMSE_OMPMMVsubAdv_gg_nn = norm(H_set(1:end)-H_set_OMPMMVsubAdv(1:end))^2/norm(H_set(1:end))^2;
        NMSE_OMPMMVsubAdv_set(ptx,nn_s) = NMSE_OMPMMVsubAdv_gg_nn;
        % 
        %% Oracle_LS_AUD_CE
        H_set_Oracle_LS = A20_Oracle_LS_AUD_CE(Y_MMV_set, Z_MMV_set, K, Ka, AUD_Index_K_set);
        % NMSE
        NMSE_Oracle_LS_gg_nn = norm(H_set(1:end)-H_set_Oracle_LS(1:end))^2/norm(H_set(1:end))^2;
        NMSE_Oracle_LS_set(ptx,nn_s) = NMSE_Oracle_LS_gg_nn;

        fprintf('Sim=%d, Ptx=%d\n',nn_s, Ptx_set(ptx));
        fprintf('MMV_OMP_A, AUD%f, CE%f\n',Pe_BOMP_gg_nn, NMSE_BOMP_gg_nn);
        fprintf('OMPMMVsubA, AUD%f, CE%f\n',Pe_OMPMMVsubA_gg_nn, NMSE_OMPMMVsubA_gg_nn);
        fprintf('OMPMMVsubAdv, AUD%f, CE%f\n',Pe_OMPMMVsubAdv_gg_nn, NMSE_OMPMMVsubAdv_gg_nn);
        fprintf('BSAMP, AUD%f, CE%f\n',Pe_BSAMP_gg_nn, NMSE_BSAMP_gg_nn);
        fprintf('BSP, AUD%f, CE%f\n',Pe_BSP_gg_nn, NMSE_BSP_gg_nn);
        % fprintf('Oracle-LS-a, CE%f\n',NMSE_Oracle_LS_A);
        fprintf('Oracle-LS, CE%f\n',NMSE_Oracle_LS_gg_nn);
        fprintf('-------------------------------------\n');

        %% Display
        % if mod(nn_s,2) == 0
        %     disp(['  G_ofdm = ' num2str(G_ofdm) ' , num_sim = ' num2str(nn_s) ' , Ptx = ' num2str(Ptx_dBm) ' dBm'])
        % end
    end
    toc
    if mod(nn_s,1==0)
        %
        Pe_BSAMP = sum(Pe_BSAMP_set,2)/nn_s;
        NMSE_BSAMP_dB = 10*log10(sum(NMSE_BSAMP_set,2)/nn_s);
        %
        Pe_BSP = sum(Pe_BSP_set,2)/nn_s;
        NMSE_BSP_dB = 10*log10(sum(NMSE_BSP_set,2)/nn_s);
        %
        Pe_BOMP = sum(Pe_BOMP_set,2)/nn_s;
        NMSE_BOMP_dB = 10*log10(sum(NMSE_BOMP_set,2)/nn_s);
        %
        Pe_MMV_BSOMP_Ad = sum(Pe_MMV_BSOMP_Ad_set,2)/nn_s;
        NMSE_MMV_BSOMP_Ad_dB = 10*log10(sum(NMSE_MMV_BSOMP_Ad_set,2)/nn_s);
        %
        Pe_MMV_BSOMP = sum(Pe_MMV_BSOMP_set,2)/nn_s;
        NMSE_MMV_BSOMP_dB = 10*log10(sum(NMSE_MMV_BSOMP_set,2)/nn_s);
        %
        Pe_GMMV_BSOMP_Ad = sum(Pe_GMMV_BSOMP_Ad_set,2)/nn_s;
        NMSE_GMMV_BSOMP_Ad_dB = 10*log10(sum(NMSE_GMMV_BSOMP_Ad_set,2)/nn_s);
        %
        Pe_GMMV_BSOMP = sum(Pe_GMMV_BSOMP_set,2)/nn_s;
        NMSE_GMMV_BSOMP_dB = 10*log10(sum(NMSE_GMMV_BSOMP_set,2)/nn_s);
        %
        NMSE_Oracle_LS_dB = 10*log10(sum(NMSE_Oracle_LS_set,2)/nn_s);

        Pe_OMPMMVsubA = sum(Pe_OMPMMVsubA_set,2)/nn_s;
        NMSE_OMPMMVsubA_dB = 10*log10(sum(NMSE_OMPMMVsubA_set,2)/nn_s);

        Pe_OMPMMVsubAdv = sum(Pe_OMPMMVsubAdv_set,2)/nn_s;
        NMSE_OMPMMVsubAdv_dB = 10*log10(sum(NMSE_OMPMMVsubAdv_set,2)/nn_s);
        %

        % save Near_Field_Pe_NMSE_G50_Ptx_10_35_v1.mat nn_s Ptx_set G_ofdm Pe_BSAMP NMSE_BSAMP_dB Pe_BSP NMSE_BSP_dB Pe_BOMP NMSE_BOMP_dB...
        %     Pe_MMV_BSOMP_Ad NMSE_MMV_BSOMP_Ad_dB Pe_MMV_BSOMP NMSE_MMV_BSOMP_dB Pe_GMMV_BSOMP_Ad NMSE_GMMV_BSOMP_Ad_dB...
        %     Pe_GMMV_BSOMP NMSE_GMMV_BSOMP_dB NMSE_Oracle_LS_dB
        save MMV_AUD_CE_Ptx.mat M_suba N_sub FacSp fs Ka awgn_en Ptx_set G_ofdm Pe_BSAMP Pe_BSP Pe_BOMP Pe_OMPMMVsubA Pe_OMPMMVsubAdv...
            NMSE_BSAMP_dB NMSE_OMPMMVsubAdv_dB NMSE_BSP_dB NMSE_BOMP_dB NMSE_OMPMMVsubA_dB NMSE_Oracle_LS_dB

    end
end
% A0_closematlabpool
disp('Finished all');

%% Plot
%% Plot
MarkerSize = 6;
LineWidth = 2;
Fontsize = 12;

figure
semilogy(Ptx_set,Pe_BSAMP,'-rx','MarkerSize',MarkerSize,'LineWidth',LineWidth); hold on; grid on;
semilogy(Ptx_set,Pe_BSP,'-bo','MarkerSize',MarkerSize,'LineWidth',LineWidth);
semilogy(Ptx_set,Pe_BOMP,'-g*','MarkerSize',MarkerSize,'LineWidth',LineWidth);
semilogy(Ptx_set,Pe_OMPMMVsubA,'-m^','MarkerSize',MarkerSize,'LineWidth',LineWidth);
semilogy(Ptx_set,Pe_OMPMMVsubAdv,'-yv','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(Ptx_set,Pe_GMMV_BSOMP_Ad,'-kp','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% semilogy(Ptx_set,Pe_GMMV_BSOMP,'-cd','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% axis([Ptx_set(1) Ptx_set(end) 1e-5 1e-1])
xlabel('Ptx [dBm]','Fontsize',Fontsize),ylabel('{\it{P_e}}','Fontsize',Fontsize)
set(gca,'FontSize',Fontsize, 'linewidth',1.5);
set(gca, 'GridLineStyle', '-.');
set(gcf, 'position', [200 300 650 550]);
h1 = legend('BSAMP','BSP','BOMP','MMV-BSOMP Adaptive','MMV-BSOMP','Location','southwest');
% h1 = legend('BSAMP','BSP','BOMP','MMV-BSOMP Adaptive','MMV-BSOMP','GMMV-BSOMP Adaptive','GMMV-BSOMP','Location','southwest');
set(h1,'Fontsize', Fontsize);

figure
plot(Ptx_set,NMSE_BSAMP_dB,'-rx','MarkerSize',MarkerSize,'LineWidth',LineWidth); hold on; grid on;
plot(Ptx_set,NMSE_BSP_dB,'-bo','MarkerSize',MarkerSize,'LineWidth',LineWidth);
plot(Ptx_set,NMSE_BOMP_dB,'-g*','MarkerSize',MarkerSize,'LineWidth',LineWidth);
plot(Ptx_set,NMSE_OMPMMVsubA_dB,'-m^','MarkerSize',MarkerSize,'LineWidth',LineWidth);
plot(Ptx_set,NMSE_OMPMMVsubAdv_dB,'-yv','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% plot(Ptx_set,NMSE_GMMV_BSOMP_Ad_dB,'-kp','MarkerSize',MarkerSize,'LineWidth',LineWidth);
% plot(Ptx_set,NMSE_GMMV_BSOMP_dB,'-cd','MarkerSize',MarkerSize,'LineWidth',LineWidth);
plot(Ptx_set,NMSE_Oracle_LS_dB,'-','Color',[0.3 0.5 0.9],'MarkerSize',MarkerSize,'LineWidth',LineWidth);
% axis([Ptx_set(1) Ptx_set(end) -50 10])
xlabel('Ptx [dBm]','Fontsize',Fontsize),ylabel('NMSE [dB]','Fontsize',Fontsize)
set(gca,'FontSize',Fontsize, 'linewidth',1.5);
set(gca, 'GridLineStyle', '-.');
set(gcf, 'position', [900 300 650 550]);
h1 = legend('BSAMP','BSP','BOMP','MMV-BSOMP Adaptive','MMV-BSOMP','Location','southwest');
% h2 = legend('BSAMP','BSP','BOMP','MMV-BSOMP Adaptive','MMV-BSOMP','GMMV-BSOMP Adaptive','GMMV-BSOMP','Oracle LS','Location','southwest');
set(h2,'Fontsize', Fontsize);
