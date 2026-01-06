% Time-invariant(TI) channel estimation for Radar with Sparse Rx array.
clear
warning on
pure_delay_en = 0;
pure_angle_en = 0;
methodImp_en = 1;
noise_en = 1;
LSF_en = 1;
 %% System Setting
Ntx = [16,1];16*ones(1,2);   % [Nx, Ny]
Nrx = [8,1];16*ones(1,2);
d_Tx_norm = 0.5;
d_Rx_norm_set = 0.5:0.5:1;  % Sparse array with larger antenna spacing
N_T = Ntx(1)*Ntx(2);
N_R = Nrx(1)*Nrx(2);
N_RF_T = 4;
N_RF_R = N_R;

c = 3e8;    % Lightspeed, [m/s]
fc = 77e9;  % carrier frequency, [Hz]
lambda = c / fc;
BW = 200e6 ;    % [Hz]
Ts = 1/BW;  % [sec]
if(LSF_en)
%     Dis_FF_min = max(2*max(Ntx)^2*d_Tx_norm^2*lambda,2*max(Nrx)^2*d_Rx_norm^2*lambda);  % min distance for far-field
    NSD = -174; % dBm/Hz
    sigma2 = 10^(NSD/10)*BW/1000;    % [watts]
    sigma = sqrt(sigma2);   
end

tau_pulse = 6*Ts;   % single-side length of pulse shaping filter
L = 32; % [sample]
tau_max = (L-1)*Ts-2*tau_pulse; % [sec]

Nc = 6; % # of clusters
Np = 15; % # of paths in each cluster
AS = 7.5; % [deg]
DS = 0.3*Ts;    % [sec]

%% CE Setting 
% CE parameter
T_RF = 30;
T_GI = 10;
P_eff = 210-T_GI;    % pilot length, P >=  L
P = P_eff+L-1;

N_redu = 1.25;
N_redu_Tx = 1;

% Monte-Carlo Simulation
iterMax = 100;
Ptx_dBm = 60;
Ptx = 10.^(Ptx_dBm/10)/1000;    % [watts]c
QuanBit = 5;
if(pure_delay_en || pure_angle_en); iterOMP = Nc; else; iterOMP_set = 100*ones(size(d_Rx_norm_set)); end
NMSE_Imp = zeros(size(d_Rx_norm_set));
NMSE_Dir = zeros(size(d_Rx_norm_set));
% startmatlabpool(2)
for iter = 1:iterMax
    %% TI Channel Setting
    if(pure_delay_en || pure_angle_en);Np = 1;end
    % Delay
    if(pure_delay_en)
        idx_perm = randperm(round(tau_max/Ts)+1)-1;
        tau(:,1) = Ts*idx_perm(1:Nc).';
    else
        tau = zeros(Nc, Np);
        for i = 1:Nc
            tau(i,1) = tau_max*rand;   % main
            tau(i,2:end) = tau(i,1)-DS+2*DS*rand(1,Np-1);
        end
    end

    % AoD
    if(pure_angle_en)
        d_test_norm = 1;
        AoD_ele = pi/2*ones(Nc,Np);
        idx_angle = 1:N_redu_Tx*Nrx(1);
        idx_angle = idx_angle(randperm(length(idx_angle)));
        AoD_azi = acos(idx_angle(1:Nc)/N_redu_Tx/Ntx(1)/d_test_norm).';
    else
        AoD_azi = zeros(Nc, Np);
        AoD_ele = zeros(Nc, Np);
        for i = 1:Nc
           AoD_azi(i,1) = AS+(360-AS)*rand;    % [deg], main
           AoD_ele(i,1) = AS+(60-AS)*rand;    % [deg], main
           AoD_azi(i,2:end) = AoD_azi(i,1)-AS+2*AS*rand(1,Np-1);   % [deg]
           AoD_ele(i,2:end) = AoD_ele(i,1)-AS+2*AS*rand(1,Np-1);   % [deg]
        end
        AoD_azi = deg2rad(AoD_azi); % [rad]
        AoD_ele = deg2rad(AoD_ele); % [rad]
    end

    % AoAs = AoDs in Radar
    AoA_azi = AoD_azi;
    AoA_ele = AoD_ele;

    % complex gain
    if(LSF_en)
        % RCSs of the targets
        RCS_min = 0.5;  % [m^2]
        RCS_max = 5;    % [m^2]
        RCS = kron(RCS_min+(RCS_max-RCS_min)*rand(Nc,1),ones(1,Np));

        % Distances of the targets
        Dis_min = 5;    % [m]
%         if(Dis_min < Dis_FF_min); error('Far-field Please!'); end
        Dis_max = 2*Dis_min;    % [m]
        Dis = kron(Dis_min+(Dis_max-Dis_min)*rand(Nc,1),ones(1,Np));

        % LSF factor
        alpha = exp(1i*2*pi*rand(Nc,Np)).*sqrt(lambda^2*RCS./(64*pi^3*Dis.^4))/sqrt(Np);
    else
        alpha = (randn(Nc*Np,1)+1i*randn(Nc*Np,1))/sqrt(2);
    end

    % Measurement matrix is block Toeplitz matrix
    P_Tplz = Blk_Tplz_Hybrid(N_T, P, L, N_RF_T, T_RF, T_GI);
    if(N_RF_R == N_R)
        Phi = kron(P_Tplz,eye(N_R));
    else
        error('Fully-digital at the Rx Please!');
    end
    
    % CE via OMP
    tic
    noise = sigma*(randn(size(P*N_R))+1i*randn(size(P*N_R)))/sqrt(2);
    for i_dRx = 1:length(d_Rx_norm_set)
        d_Rx_norm = d_Rx_norm_set(i_dRx);
        iterOMP = iterOMP_set(i_dRx);
        
        % Steering matrices
        SterMtx_Tx = SterMtx_UPA(Ntx, d_Tx_norm, AoD_azi(:), AoD_ele(:));
        SterMtx_Rx = SterMtx_UPA(Nrx, d_Rx_norm, AoA_azi(:), AoA_ele(:));

         % CIR in the spatial and the delay (SD) domains of dimensioni
         % N_R-by-L*N_Tz
        H_SD = CIR_SD_TI_new(SterMtx_Tx, SterMtx_Rx, 1:L, diag(alpha(:)), Nc*Np, tau(:), tau_pulse, Ts);

        % Rx Signals w/o noise
        Y = Phi*vec(H_SD);
    
        % Dictionary
        if Nrx(1) > 1; Grx(1) = N_redu*Nrx(1); else; Grx(1) = 1; end % Dictionary dimension
        if Nrx(2) > 1; Grx(2) = N_redu*Nrx(2); else; Grx(2) = 1; end % Dictionary dimension
        if Ntx(1) > 1; Gtx(1) = N_redu_Tx*Ntx(1); else; Gtx(1) = 1; end % Dictionary dimension
        if Ntx(2) > 1; Gtx(2) = N_redu_Tx*Ntx(2); else; Gtx(2) = 1; end % Dictionary dimension
        G_T = Gtx(1)*Gtx(2);
        G_R = Grx(1)*Grx(2);
        Ftx_1 = dftmtx(round(2*d_Tx_norm*Gtx(1)))/sqrt(Ntx(1));
        Ftx_2 = dftmtx(round(2*d_Tx_norm*Gtx(2)))/sqrt(Ntx(2));
        Atx = kron(Ftx_1(1:round(2*d_Tx_norm):round(2*d_Tx_norm*(Ntx(1)-1)+1),1:Gtx(1)),...
                   Ftx_2(1:round(2*d_Tx_norm):round(2*d_Tx_norm*(Ntx(2)-1)+1),1:Gtx(2)));
       if(2*d_Rx_norm == fix(2*d_Rx_norm))
            Frx_1 = dftmtx(round(2*d_Rx_norm*Grx(1)))/sqrt(Nrx(1));
            Frx_2 = dftmtx(round(2*d_Rx_norm*Grx(2)))/sqrt(Nrx(2));
            Arx = kron(Frx_1(1:round(2*d_Rx_norm):round(2*d_Rx_norm*(Nrx(1)-1)+1),1:Grx(1)),...
                       Frx_2(1:round(2*d_Rx_norm):round(2*d_Rx_norm*(Nrx(2)-1)+1),1:Grx(2)));
       else
           Arx = zeros(N_R,G_R);
           for g = 1:G_R
              Arx(:,g) =  exp(-1i*2*pi*(g-1)/G_R*(0:N_R-1).')/sqrt(N_R);
           end
       end
       
        Yn = sqrt(Ptx)*Y+noise_en*noise;
        NormFac = max(abs([real(Yn);imag(Yn)]));
        Yn_quan = Quan_simple(Yn/NormFac,QuanBit,'uniform');
        SenMtx = sqrt(Ptx)*Phi*kron(kron(eye(L),conj(Atx)),Arx)/NormFac;
        
        if(methodImp_en)
            % improved OMP (OMP_SR)
            Supp = zeros(iterOMP,1);
            Supp_L = zeros(iterOMP,1);
            Supp_AoD_azi = zeros(iterOMP,1);
            Supp_AoA_azi = zeros(iterOMP,1);
            Supp_idx_AoD_azi = zeros(iterOMP,1);
            Supp_idx_AoA_azi = zeros(iterOMP,1);
            y_k_com = Yn_quan(:);
            r_k_com = Yn_quan(:);

            % Delays normalized by sampling period
            idx_L = ((tau+tau_pulse)/Ts)+1; % [1,L]

            % virtual angle (or called spatial frequency)
            vA_Radar_azi = cos(AoD_azi).*sin(AoD_ele);
            vA_Radar_ele = sin(AoD_azi).*sin(AoD_ele);

            for i = 1:iterOMP
                c_k_com = SenMtx'*r_k_com;
                [~, idx_Est] = max(sum(abs(c_k_com),2));
                Supp(i) = idx_Est;
                [idx_AoA_Est,idx_LGtx_Est] = ind2sub([G_R,L*G_T],idx_Est);
                [idx_AoD_Est,idx_L_Est] = ind2sub([G_T,L],idx_LGtx_Est);
                Supp_L(i) = idx_L_Est;
                [idx_AoD_ele_Est,idx_vAoD_azi_Est] = ind2sub([Gtx(2),Gtx(1)],idx_AoD_Est);
                [idx_AoA_ele_Est,idx_vAoA_azi_Est] = ind2sub([Grx(2),Grx(1)],idx_AoA_Est);
                Supp_idx_AoD_azi(i) = idx_AoD_Est;
                Supp_idx_AoA_azi(i) = idx_AoA_Est;

                % Azi estimation
                vAoD_azi_Est = (idx_vAoD_azi_Est-1)/(N_redu_Tx*Ntx(1)*d_Tx_norm);
                vAoD_azi_Est(vAoD_azi_Est>1) = vAoD_azi_Est(vAoD_azi_Est>1) - 2;
                vAoA_azi_Est = (idx_vAoA_azi_Est-1)/(N_redu*Nrx(1)*d_Rx_norm);
                vAoA_azi_Est = vAoA_azi_Est + round((vAoD_azi_Est-vAoA_azi_Est)*d_Rx_norm)/d_Rx_norm;
                Supp_AoD_azi(i) = vAoD_azi_Est;
                Supp_AoA_azi(i) = vAoA_azi_Est;

                % Ele estimation�������ڴ���������ʱ�����й���
                vAoA_ele_Est = -1+2*rand;

                % Adjust estimated AoD according to estimated AoA with higher resolution 
                q = sqrt(Ptx)*Phi(:,(idx_L_Est-1)*N_T*N_R+1:idx_L_Est*N_T*N_R)*kron(kron(exp(1i*2*pi*d_Tx_norm*vAoA_azi_Est*(0:Ntx(1)-1).'),...
                                        exp(1i*2*pi*d_Tx_norm*vAoA_ele_Est*(0:Ntx(2)-1).'))/sqrt(N_T),Arx(:,idx_AoA_Est))/NormFac;
                SenMtx_supp = [SenMtx(:,Supp(1:i-1)),q];
%                 if(rank(SenMtx_supp) == min(size(SenMtx_supp)))
                    SenMtx(:,idx_Est) = q;
%                 end
                warning off
                xi_hat_com = SenMtx(:,Supp(1:i))\y_k_com;alpha;
                warning on
                r_k_com = y_k_com - SenMtx(:,Supp(1:i))*xi_hat_com;
            end
            H_SD_hat_Rec = CIR_SD_TI_new(exp(-1i*2*pi*d_Tx_norm*(0:Ntx(1)-1).'*Supp_AoA_azi.')/sqrt(N_T),...
                                            exp(-1i*2*pi*d_Rx_norm*(0:Nrx(1)-1).'*Supp_AoA_azi.')/sqrt(N_R), ...
                                                1:L, diag(xi_hat_com), length(xi_hat_com), (Supp_L-1)*Ts-tau_pulse, tau_pulse, Ts);
            NMSE_temp_Rec = norm(H_SD_hat_Rec - H_SD,'fro')^2/norm(H_SD,'fro')^2;
            NMSE_Imp(i_dRx) = NMSE_Imp(i_dRx) + NMSE_temp_Rec;

            disp(['  Ptx = ' num2str(Ptx_dBm) ' dBm, iterOMP = ' num2str(iterOMP)...
              ', QuanBit = ' num2str(QuanBit) ', iter = ' num2str(iter) '/'  num2str(iterMax)...
             ', N_redu = ' num2str(N_redu) ', d_Rx_norm = ' num2str(d_Rx_norm)...
             ', NMSE_Imp_now = ' num2str(10*log10(NMSE_temp_Rec)) ' dB, NMSE_Imp_avg = ' num2str(10*log10(NMSE_Imp(i_dRx)/iter)) ' dB.']);
        else
            % OMP directly
            h_AD_hat_Dir = S_OMP_Algorithm(Yn_quan(:), SenMtx, -1, iterOMP);
            H_AD_hat_Dir = reshape(h_AD_hat_Dir,G_R,L*G_T);
            H_SD_hat_Dir = Arx*H_AD_hat_Dir*kron(eye(L),Atx');
            NMSE_temp_Dir = norm(H_SD_hat_Dir - H_SD,'fro')^2/norm(H_SD.','fro')^2;
            NMSE_Dir(i_dRx) = NMSE_Dir(i_dRx) + NMSE_temp_Dir;

            disp(['  Ptx = ' num2str(Ptx_dBm) ' dBm, iterOMP = ' num2str(iterOMP)...
                      ', QuanBit = ' num2str(QuanBit) ', iter = ' num2str(iter) '/'  num2str(iterMax)...
                     ', N_redu = ' num2str(N_redu) ', d_Rx_norm = ' num2str(d_Rx_norm)...
                     ', NMSE_Dir_now = ' num2str(10*log10(NMSE_temp_Dir)) ' dB, NMSE_Dir_avg = ' num2str(10*log10(NMSE_Dir(i_dRx)/iter)) ' dB']);
        end
    end
    toc
end
% closematlabpool
NMSE_dB = 10*log10(NMSE_Imp/iterMax)