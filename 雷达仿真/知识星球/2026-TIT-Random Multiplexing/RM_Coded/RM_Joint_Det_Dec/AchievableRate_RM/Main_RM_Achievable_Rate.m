%% Parameter Initialization
clc; clear; close all;
%% Monte Carlo SE 
Num_MC=1;
fs_N=1;
delta_f=15000;

Ns=4; % Transmitting Antenna
Nr=4; % Receiving Antenna

N =8;
delta =1;
M = 32;
fs=fs_N*M*delta_f;
MN=M*N;
rho=0.6;
P = 5;% Number of Path
index_D = 1;  %1 means has Doppler, 0 means no Doppler
car_fre = 4*10^9; % Carrier frequency
max_speed=300; 
dop = max_speed*(1e3/3600)*(car_fre/3e8);  % maximum Doppler, 150km/h as an example
beta=0.4;
snr_dB =[-10:0.5:30];
snr=10.^(snr_dB/10);
v_LE    =[1e-5:1e-4:0.1, 0.1:0.01:1];% 
RHO_LE_RM=zeros(length(snr_dB),length(v_LE));

%% RM
orderm=randperm(N*M,N*M);
Em=eye(N*M);
Sm=Em(orderm,:);
A1=dftmtx(N*M)/sqrt(N*M)*Sm; 
A1_nr=kron(eye(Nr),A1);
A1_ns=kron(eye(Ns),A1');
for j=1:Num_MC
    j
    %% MIMO channel model       
    A=zeros(Nr,Ns,P);
    for i=1:P
    A(:,:,i)=relatedh(Nr,Ns,rho,rho);
    end

    H_TD_ce=cell(Ns,Nr);
    for ns=1:Ns
        for nr=1:Nr
        [~,g_m_0] = getchannel_related(M,N,delta_f,fs,fs_N,zeros(M*N,1),P,index_D,dop,beta,A(nr,ns,:));
        L=size(g_m_0, 2)-1;         
        H_T=zeros(N*M, N*M);        % Time-domain channel matrix MN*MN 
        for aa=1:N*M
            temp=g_m_0(aa,:);
            temp1=fliplr(temp);
            temp2=[temp1(end),zeros(1,N*M-L-1),temp1(1:end-1)];
            H_T(aa,:)=circshift(temp2,aa-1,2);
        end
           H_TD_ce{nr,ns}=H_T;
        end
    end
    H_TD_m=cell2mat(H_TD_ce);% %Time domain channel
    [~,H_TD_m]=channel_norm(H_TD_m);
    H_DD_m_RM=A1_nr*H_TD_m*A1_ns;
    [~, Dia_RM, ~] = svd(H_DD_m_RM);
    dia_RM = diag(Dia_RM);
    dia_RM = dia_RM.';
    % system parametersdd
    Sigma_n= snr.^-1;
    %%  SE
    vpre_LE=[1e-5:1e-4:0.1, 0.1:0.01:1]; 
    % %vpost_LE=GOAMP_GLE_Clip_SE(delta, clip, snr.^-1, vpre_LE);


    vpost_LE_RM= LE_OAMP_SE(vpre_LE, dia_RM, Sigma_n, Nr*MN, Ns*MN);

    for vid = 1:length(snr_dB)
        v_mmse(vid,:) = GOAMP_GLE_inv_SE(v_LE, vpre_LE, vpost_LE_RM(vid,:));
        rho_LE(vid,:) = 1./v_LE- 1./v_mmse(vid,:);
        RHO_LE_RM(vid,:)= RHO_LE_RM(vid,:)+rho_LE(vid,:);
    end
end
%% Achievable Rate
RHO_LE_RM=RHO_LE_RM./Num_MC;
PHI_1_RM =RHO_LE_RM(:,end);
SNR_end_RM = RHO_LE_RM(:,1);

fprintf('OAMP Gaussian \n');
[R_OAMP_Gaussian_RM, R_Sep_Gaussian_RM]= Rate_OAMP_Gaussian(snr,v_LE, RHO_LE_RM, PHI_1_RM,SNR_end_RM);  %计算面积

fprintf('OAMP QPSK \n');
[R_OAMP_QPSK_RM, R_Sep_QPSK_RM] = Rate_OAMP_QPSK(snr,v_LE, RHO_LE_RM, PHI_1_RM,SNR_end_RM);

fprintf('OAMP 8PSK \n');
[R_OAMP_8PSK_RM, R_Sep_8PSK_RM]  = Rate_OAMP_8PSK(snr,v_LE, RHO_LE_RM, PHI_1_RM,SNR_end_RM);  

fprintf('OAMP 16QAM \n');
[R_OAMP_16QAM_RM,  R_Sep_16QAM_RM]= Rate_OAMP_16QAM(snr,v_LE, RHO_LE_RM, PHI_1_RM,SNR_end_RM);
 
%% figure
close all
YMatrix3=[R_OAMP_Gaussian_RM; R_OAMP_QPSK_RM; R_OAMP_8PSK_RM; R_OAMP_16QAM_RM];
YMatrix4=[R_Sep_Gaussian_RM; R_Sep_QPSK_RM; R_Sep_8PSK_RM; R_Sep_16QAM_RM];
figure(1)
axes1 = axes;
hold(axes1,'on');
plot3 = plot(snr_dB,YMatrix3,'LineWidth',1,'Color',[1 0 0]);
set(plot3(1),'DisplayName','RM-R_{MAMP-Gaussian}');
set(plot3(2),'DisplayName','RM-R_{MAMP-QPSK}','Marker','*');
set(plot3(3),'DisplayName','RM-R_{MAMP-8PSK}','Marker','s');
set(plot3(4),'DisplayName','RM-R_{MAMP-16QAM}','Marker','d');

plot4 = plot(snr_dB,YMatrix4,'LineWidth',1,'LineStyle','--','Color',[0 0 1]);
set(plot4(1),'DisplayName','RM-R_{Sep-Gaussian}');
set(plot4(2),'DisplayName','RM-R_{Sep-QPSK}','Marker','*');
set(plot4(3),'DisplayName','RM-R_{Sep-8PSK}','Marker','s');
set(plot4(4),'DisplayName','RM-R_{Sep-16QAM}','Marker','d');

title({['delta=',num2str(delta),'N=',num2str(N),',M=',num2str(M)]});
xlabel('SNR (dB)');
ylabel('Rate (bits)');
% xlim([min(snr_dB) max(snr_dB)]);
% ylim([0,5.5]);

set(axes1,'FontSize',15);
% legend
legend1 = legend(axes1,'show');
