% Time-invariant(TI) channel estimation for Radar with Sparse Rx array.
clear
warning on
pure_delay_en = 0;
pure_angle_en = 0;
noise_en = 1;
LSF_en = 1;

methodImp_en = 1;
methodDir_en = 1;
methodBomp_en = 1;
methodLS_en = 0;
methodOLS_en = 0;
plot_en = 0;
AD_predefined_en = 1;

 %% System Setting
Ntx = [16,1];16*ones(1,2);   % [Nx, Ny]
Nrx = [8,1];16*ones(1,2);
d_Tx_norm = 0.5;
d_Rx_norm = 1.5;  % Sparse array with lager antenna spacing
N_T = Ntx(1)*Ntx(2);
N_R = Nrx(1)*Nrx(2);
N_RF_T = 4;
QuanBit = 5;

c = 3e8;    % Lightspeed, [m/s]
fc = 77e9;  % carrier frequency, [Hz]
lambda = c / fc;
BW = 200e6 ;    % [Hz]
Ts = 1/BW;  % [sec]
if(LSF_en)
    Dis_FF_min = max(2*max(Ntx)^2*d_Tx_norm^2*lambda,2*max(Nrx)^2*d_Rx_norm^2*lambda);  % min distance for far-field
    NSD = -174; % dBm/Hz
    sigma2 = 10^(NSD/10)*BW/1000;    % [watts]
    sigma = sqrt(sigma2);
end

tau_pulse = 6*Ts;   % single-side length of pulse shaping filter
L = 32; % [sample]
tau_max = 20/c;(L-1)*Ts-2*tau_pulse; % [sec]

Nc = 6; % # of clusters
Np = 1; % # of paths in each cluster
AS = 7.5; % [deg]
DS = 0.3*Ts;    % [sec]

%% CE Setting
% Dictionary
N_redu = 2;
N_redu_Tx = 1;
if Nrx(1) > 1; Grx(1) = N_redu*Nrx(1); else; Grx(1) = 1; end % Dictionary dimension
if Nrx(2) > 1; Grx(2) = N_redu*Nrx(2); else; Grx(2) = 1; end % Dictionary dimension
if Ntx(1) > 1; Gtx(1) = N_redu_Tx*Ntx(1); else; Gtx(1) = 1; end % Dictionary dimension
if Ntx(2) > 1; Gtx(2) = N_redu_Tx*Ntx(2); else; Gtx(2) = 1; end % Dictionary dimension
% Gtx = Ntx;
G_T = Gtx(1)*Gtx(2);
G_R = Grx(1)*Grx(2);
Ftx_1 = dftmtx(round(2*d_Tx_norm*Gtx(1)))/sqrt(Ntx(1));
Ftx_2 = dftmtx(round(2*d_Tx_norm*Gtx(2)))/sqrt(Ntx(2));
Atx = kron(Ftx_1(1:round(2*d_Tx_norm):round(2*d_Tx_norm*(Ntx(1)-1)+1),1:Gtx(1)),...
           Ftx_2(1:round(2*d_Tx_norm):round(2*d_Tx_norm*(Ntx(2)-1)+1),1:Gtx(2)));
Frx_1 = dftmtx(round(2*d_Rx_norm*Grx(1)))/sqrt(Nrx(1));
Frx_2 = dftmtx(round(2*d_Rx_norm*Grx(2)))/sqrt(Nrx(2));
Arx = kron(Frx_1(1:round(2*d_Rx_norm):round(2*d_Rx_norm*(Nrx(1)-1)+1),1:Grx(1)),...
           Frx_2(1:round(2*d_Rx_norm):round(2*d_Rx_norm*(Nrx(2)-1)+1),1:Grx(2)));
       
% CE parameter
T_RF = 30;
T_GI = 10;
P_eff = 300-T_GI;    % pilot length, P >=  L
P = P_eff+L-1;

% Monte-Carlo Simulation
if(plot_en); iterMax = 1; else; iterMax = 100; end
Ptx_dBm = 60;
Ptx = 10.^(Ptx_dBm/10)/1000;    % [watts]
if(pure_delay_en && pure_angle_en); iterOMP = Nc; else; iterOMP = 100; end
if(Np == 1); iterOMP = 15; end
if(methodImp_en); NMSE_Imp = zeros(size(Ptx)); end
if(methodDir_en); NMSE_Dir = zeros(size(Ptx)); end
if(methodBomp_en); NMSE_Bomp = zeros(size(Ptx)); end
if(methodLS_en); NMSE_LS = zeros(size(Ptx)); end
if(methodOLS_en); NMSE_OLS = zeros(size(Ptx)); end
% startmatlabpool(2)
tau_predefine = tau_max*[0.12;0.31;0.55;0.9;0.78;0.8];
AoD_azi_predefine = 180*(rand(Nc,1)>0.5);
AoD_ele_predefine = 90*[0.51;0.24;0.33;0.75;0.78;0.5];
for iter = 1:iterMax
    %% TI Channel Setting
    if(pure_delay_en && pure_angle_en);Np = 1;end
    % Delay
    if(pure_delay_en)
        idx_perm = randperm(round(tau_max/Ts)+1)-1;
        tau = kron(Ts*idx_perm(1:Nc).',ones(1,Np));
    else
        tau = zeros(Nc, Np);
        for i = 1:Nc
            if(AD_predefined_en)
                tau(i,1) = tau_predefine(i);
            else
                tau(i,1) = tau_max*rand;   % main
            end
            tau(i,2:end) = tau(i,1)-DS+2*DS*rand(1,Np-1);
        end
    end

    % AoD
    if(pure_angle_en)
        d_test_norm = 1.5;
        AoD_azi = zeros(Nc,Np);
        idx_angle = 1:N_redu*Nrx(1);
        idx_angle = idx_angle(randperm(length(idx_angle)));
        AoD_ele = kron(asin(idx_angle(1:Nc)/N_redu/Ntx(1)/d_test_norm).',ones(1,Np));
    else
        AoD_azi = zeros(Nc, Np);
        AoD_ele = zeros(Nc, Np);
        for i = 1:Nc
            if(AD_predefined_en)
                AoD_azi(i,1) = AoD_azi_predefine(i);    % [deg], main
                AoD_ele(i,1) = AoD_ele_predefine(i);    % [deg], main
            else
                AoD_azi(i,1) = AS+(360-AS)*rand;    % [deg], main
                AoD_ele(i,1) = AS+(60-AS)*rand;    % [deg], main
            end          
           AoD_azi(i,2:end) = AoD_azi(i,1)-AS+2*AS*rand(1,Np-1);   % [deg]
           AoD_ele(i,2:end) = AoD_ele(i,1)-AS+2*AS*rand(1,Np-1);   % [deg]
        end
        AoD_azi = deg2rad(AoD_azi); % [rad]
        AoD_ele = deg2rad(AoD_ele); % [rad]
    end

    % AoAs = AoDs in Radar
    AoA_azi = AoD_azi;
    AoA_ele = AoD_ele;
    
    % Delays normalized by sampling period
    idx_L = ((tau+tau_pulse)/Ts)+1; % [1,L]

    % virtual angle (or called spatial frequency)
    vA_Radar_azi = cos(AoD_azi).*sin(AoD_ele);
    vA_Radar_ele = sin(AoD_azi).*sin(AoD_ele);
    
    % Plot
    if(plot_en)
        figure
        subplot(211)
        for i=1:Nc
            polarplot(asin(vA_Radar_azi(i,:))+pi/2, tau(i,:)*c/2,'o','Markersize',10)   %��pi/2����ͼ���ƶ����ϰ�Բ
            hold on
        end
        ax = gca;
        axis tight
        ax.ThetaLim = [0 180];
        ax.RLim = [0 10];
        ax.ThetaTickLabel = {'$-90^\circ$','$-60^\circ$','$-30^\circ$','$0^\circ$','$30^\circ$','$60^\circ$','${90^\circ }$'};
        ax.TickLabelInterpreter = 'latex';
        title({'Radar Target Estimation (Angles and Ranges)','without angular ambiguity'})
    end

    % complex gain
    if(LSF_en)
        % RCSs of the targets
        RCS_min = 0.5;  % [m^2]
        RCS_max = 5;    % [m^2]
        RCS = kron(RCS_min+(RCS_max-RCS_min)*rand(Nc,1),ones(1,Np));

        % Distances of the targets
        Dis_min = 5;    % [m]
        if(Dis_min < Dis_FF_min); error('Far-field Please!'); end
        Dis_max = 2*Dis_min;    % [m]
        Dis = kron(Dis_min+(Dis_max-Dis_min)*rand(Nc,1),ones(1,Np));

        % LSF factor
        alpha = exp(1i*2*pi*rand(Nc,Np)).*sqrt(lambda^2*RCS./(64*pi^3*Dis.^4))/sqrt(Np);
    else
        alpha = (randn(Nc*Np,1)+1i*randn(Nc*Np,1))/sqrt(2);
    end

    % Measurement matrix is block Toeplitz matrix
    P_Tplz = Blk_Tplz_Hybrid(N_T, P, L, N_RF_T, T_RF, T_GI);
    Phi = kron(P_Tplz,eye(N_R));
    if(methodLS_en); Phi_LS = kron(Blk_Tplz_Hybrid(N_T, L*N_T, L, N_RF_T, T_RF, T_GI),eye(N_R)); end
    
    % Steering matrices
    SterMtx_Tx = SterMtx_UPA(Ntx, d_Tx_norm, AoD_azi(:), AoD_ele(:));
    SterMtx_Rx = SterMtx_UPA(Nrx, d_Rx_norm, AoA_azi(:), AoA_ele(:));

     % CIR in the spatial and the delay (SD) domains of dimensioni
     % N_R-by-L*N_Tz
    H_SD = CIR_SD_TI_new(SterMtx_Tx, SterMtx_Rx, 1:L, diag(alpha(:)), Nc*Np, tau(:), tau_pulse, Ts);
    
    % Rx Signals w/o noise
    Y = Phi*vec(H_SD);
    
    % CE via OMP
    tic
    noise = sigma*(randn(size(Y))+1i*randn(size(Y)))/sqrt(2);
    for i_Ptx = 1:length(Ptx)
        Yn = sqrt(Ptx(i_Ptx))*Y+noise_en*noise;
        NormFac = max(abs([real(Yn);imag(Yn)]));
        Yn_quan = Quan_simple(Yn/NormFac,QuanBit,'uniform');
        
        if(methodImp_en)
            SenMtx = sqrt(Ptx(i_Ptx))*Phi*kron(kron(eye(L),conj(Atx)),Arx)/NormFac;
            % improved OMP
            Supp = zeros(iterOMP,1);
            Supp_L = zeros(iterOMP,1);
            Supp_AoD_azi = zeros(iterOMP,1);
            Supp_AoA_azi = zeros(iterOMP,1);
            Supp_idx_AoD_azi = zeros(iterOMP,1);
            Supp_idx_AoA_azi = zeros(iterOMP,1);
            y_k_com = Yn_quan(:);
            r_k_com = Yn_quan(:);
            Supp_AoA_azi_Amb = zeros(iterOMP,1);

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
                Supp_AoA_azi_Amb(i) = vAoA_azi_Est;
                vAoA_azi_Est = vAoA_azi_Est + round((vAoD_azi_Est-vAoA_azi_Est)*d_Rx_norm)/d_Rx_norm;
                Supp_AoD_azi(i) = vAoD_azi_Est;
                Supp_AoA_azi(i) = vAoA_azi_Est;

                % Ele estimation�������ڴ���������ʱ�����й���
                vAoA_ele_Est = -1+2*rand;

                % Adjust estimated AoD according to estimated AoA with higher resolution 
                q = sqrt(Ptx(i_Ptx))*Phi(:,(idx_L_Est-1)*N_T*N_R+1:idx_L_Est*N_T*N_R)*kron(kron(exp(1i*2*pi*d_Tx_norm*vAoA_azi_Est*(0:Ntx(1)-1).'),...
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
            NMSE_Imp(i_Ptx) = NMSE_Imp(i_Ptx) + NMSE_temp_Rec;
            
            H_SD_hat_Amb = CIR_SD_TI_new(exp(-1i*2*pi*d_Tx_norm*(0:Ntx(1)-1).'*Supp_AoD_azi.')/sqrt(N_T),...
                                exp(-1i*2*pi*d_Rx_norm*(0:Nrx(1)-1).'*Supp_AoA_azi_Amb.')/sqrt(N_R), ...
                                    1:L, diag(xi_hat_com), length(xi_hat_com), (Supp_L-1)*Ts-tau_pulse, tau_pulse, Ts);
            NMSE_temp_Amb = norm(H_SD_hat_Amb - H_SD,'fro')^2/norm(H_SD,'fro')^2;
            
            if(plot_en)        
                subplot(211)
                polarplot(asin(Supp_AoA_azi(abs(xi_hat_com)>sigma/sqrt(Nc*Np)))+pi/2, ((Supp_L(abs(xi_hat_com)>sigma/sqrt(Nc*Np))-1)*Ts-tau_pulse)*c/2,'kx')   %��pi/2����ͼ���ƶ����ϰ�Բ
                h = legend('Target 1','Target 2','Target 3','Target 4','Target 5','Target 6','Estimation Results');
                set(h,'Interpreter','latex');
                
                subplot(223)
%                 MYpcolor(abs(Arx'*H_SD*kron(eye(L),Atx)))
                MYpcolor(abs(H_SD))
                h = title({'Heat map of the real radar CIR','(perfect recovery)'});
%                 set(h,'Interpreter','latex');
                subplot(224)
                MYpcolor(abs(H_SD_hat_Rec))
                h = title({'Heat map of the estimated radar CIR',['NMSE = ' num2str(10*log10(NMSE_temp_Rec)) 'dB']});
%                 set(h,'Interpreter','latex');

                figure
                subplot(211)
                for i=1:Nc
                    polarplot(asin(vA_Radar_azi(i,:))+pi/2, tau(i,:)*c/2,'o','Markersize',10)   %��pi/2����ͼ���ƶ����ϰ�Բ
                    hold on
                end
                ax = gca;
                axis tight
                ax.ThetaLim = [0 180];
                ax.RLim = [0 10];
                ax.ThetaTickLabel = {'$-90^\circ$','$-60^\circ$','$-30^\circ$','$0^\circ$','$30^\circ$','$60^\circ$','${90^\circ }$'};
                ax.TickLabelInterpreter = 'latex';
                title({'Radar Target Estimation (Angles and Ranges)','with angular ambiguity'})
                
                polarplot(asin(Supp_AoA_azi_Amb(abs(xi_hat_com)>sigma/sqrt(Nc*Np)))+pi/2, ((Supp_L(abs(xi_hat_com)>sigma/sqrt(Nc*Np))-1)*Ts-tau_pulse)*c/2,'kx')   %��pi/2����ͼ���ƶ����ϰ�Բ
                h = legend('Target 1','Target 2','Target 3','Target 4','Target 5','Target 6','Estimation Results');
                set(h,'Interpreter','latex');
                
                subplot(223)
                MYpcolor(abs(H_SD))
                h = title({'Heat map of the real radar CIR','(perfect recovery)'});
%                 set(h,'Interpreter','latex');
                subplot(224)
                MYpcolor(abs(H_SD_hat_Amb))
                h = title({'Heat map of the estimated radar CIR',['NMSE = ' num2str(10*log10(NMSE_temp_Amb)) 'dB']});
%                 set(h,'Interpreter','latex');
            end
            
            disp(['  OMP-SR, Ptx = ' num2str(Ptx_dBm(i_Ptx)) ' dBm, iterOMP = ' num2str(iterOMP)...
              ', QuanBit = ' num2str(QuanBit) ', iter = ' num2str(iter) '/'  num2str(iterMax)...
             ', N_redu = ' num2str(N_redu) ', d_Rx_norm = ' num2str(d_Rx_norm)...
             ', NMSE_Imp_now = ' num2str(10*log10(NMSE_temp_Rec)) 'dB, NMSE_Imp_avg = ' num2str(10*log10(NMSE_Imp(i_Ptx)/iter)) ' dB.']);
        end
        
        if(methodDir_en)
            SenMtx = sqrt(Ptx(i_Ptx))*Phi*kron(kron(eye(L),conj(Atx)),Arx)/NormFac;
            % OMP directly
            h_AD_hat_Dir = S_OMP_Algorithm(Yn_quan(:), SenMtx, -1, iterOMP);
            H_AD_hat_Dir = reshape(h_AD_hat_Dir,G_R,L*G_T);
            H_SD_hat_Dir = Arx*H_AD_hat_Dir*kron(eye(L),Atx');
            NMSE_temp_Dir = norm(H_SD_hat_Dir - H_SD,'fro')^2/norm(H_SD.','fro')^2;
            NMSE_Dir(i_Ptx) = NMSE_Dir(i_Ptx) + NMSE_temp_Dir;

            disp(['  DirOMP, Ptx = ' num2str(Ptx_dBm(i_Ptx)) ' dBm, iterOMP = ' num2str(iterOMP)...
                      ', QuanBit = ' num2str(QuanBit) ', iter = ' num2str(iter) '/'  num2str(iterMax)...
                     ', N_redu = ' num2str(N_redu) ', d_Rx_norm = ' num2str(d_Rx_norm)...
                     ', NMSE_Dir_now = ' num2str(10*log10(NMSE_temp_Dir)) 'dB, NMSE_Dir_avg = ' num2str(10*log10(NMSE_Dir(i_Ptx)/iter)) ' dB']);
        end
        
        if(methodBomp_en)
            % BOMP
            [h_AD_hat_Bomp,~,Supp_Bomp,idx_L_est] = B_OMP_Algorithm(Yn_quan(:), sqrt(Ptx(i_Ptx))/NormFac*Phi, N_R*N_T, -1, 10, Ts, tau_pulse);
            H_SD_hat_Bomp = reshape(h_AD_hat_Bomp,N_R,L*N_T);
            NMSE_temp_Bomp = norm(H_SD_hat_Bomp - H_SD,'fro')^2/norm(H_SD,'fro')^2;
            NMSE_Bomp(i_Ptx) = NMSE_Bomp(i_Ptx) + NMSE_temp_Bomp;

            disp(['  BOMP, Ptx = ' num2str(Ptx_dBm(i_Ptx)) ' dBm, iterOMP = ' num2str(iterOMP)...
                      ', QuanBit = ' num2str(QuanBit) ', iter = ' num2str(iter) '/'  num2str(iterMax)...
                     ', N_redu = ' num2str(N_redu) ', d_Rx_norm = ' num2str(d_Rx_norm)...
                     ', NMSE_Dir_now = ' num2str(10*log10(NMSE_temp_Bomp)) 'dB, NMSE_Dir_avg = '...
                     num2str(10*log10(NMSE_Bomp(i_Ptx)/iter)) ' dB']);           
        end
        
        if(methodLS_en)
            Yn_LS = sqrt(Ptx(i_Ptx))*Phi_LS*vec(H_SD)+noise_en*sigma*(randn(L*N_R*N_T,1)+1i*randn(L*N_R*N_T,1))/sqrt(2);
            NormFac = max(abs([real(Yn_LS);imag(Yn_LS)]));
            Yn_quan_LS = Quan_simple(Yn_LS/NormFac,QuanBit,'uniform');
            h_SD_hat_LS = (sqrt(Ptx(i_Ptx))/NormFac*Phi_LS) \ Yn_quan_LS(:);
            NMSE_temp_LS = norm(h_SD_hat_LS - vec(H_SD),'fro')^2/norm(H_SD,'fro')^2;
            NMSE_LS(i_Ptx) = NMSE_LS(i_Ptx) + NMSE_temp_LS;
            
            disp(['  LS, Ptx = ' num2str(Ptx_dBm(i_Ptx)) ' dBm, iterOMP = ' num2str(iterOMP)...
                      ', QuanBit = ' num2str(QuanBit) ', iter = ' num2str(iter) '/'  num2str(iterMax)...
                     ', N_redu = ' num2str(N_redu) ', d_Rx_norm = ' num2str(d_Rx_norm)...
                     ', NMSE_Dir_now = ' num2str(10*log10(NMSE_temp_LS)) 'dB, NMSE_Dir_avg = ' num2str(10*log10(NMSE_LS(i_Ptx)/iter)) ' dB']);
        end
        
        if(methodOLS_en)
            % Oracle LS
            idx_zero = find(abs(vec(H_SD))==0);
            idx_Nzero = setdiff(1:L*N_R*N_T,idx_zero);
            alpha_est_OLS = (sqrt(Ptx(i_Ptx))*Phi(:,idx_Nzero)/NormFac*kron(eye(length(idx_Nzero)/(N_R*N_T)),kr(conj(SterMtx_Tx),SterMtx_Rx))) \ Yn_quan(:);
            h_est_OLS = kron(eye(length(idx_Nzero)/(N_R*N_T)),kr(conj(SterMtx_Tx),SterMtx_Rx))*alpha_est_OLS;
            h_perf = vec(H_SD); h_perf = h_perf(idx_Nzero);
            NMSE_temp_OLS = norm(h_est_OLS - h_perf,'fro')^2/norm(h_perf.','fro')^2;
            NMSE_OLS(i_Ptx) = NMSE_OLS(i_Ptx) + NMSE_temp_OLS;
            
            disp(['  OLS, Ptx = ' num2str(Ptx_dBm(i_Ptx)) ' dBm, iterOMP = ' num2str(iterOMP)...
                      ', QuanBit = ' num2str(QuanBit) ', iter = ' num2str(iter) '/'  num2str(iterMax)...
                     ', N_redu = ' num2str(N_redu) ', d_Rx_norm = ' num2str(d_Rx_norm)...
                     ', NMSE_Dir_now = ' num2str(10*log10(NMSE_temp_OLS)) 'dB, NMSE_Dir_avg = ' num2str(10*log10(NMSE_OLS(i_Ptx)/iter)) ' dB']);
        end

    end
    if(pure_delay_en || pure_angle_en || ~LSF_en || ~noise_en || QuanBit==Inf); disp('Not Practical! Data for Test or Comparison ONLY'); end
    toc
end
% closematlabpool
% NMSE_dB = 10*log10(NMSE_Imp/iterMax)