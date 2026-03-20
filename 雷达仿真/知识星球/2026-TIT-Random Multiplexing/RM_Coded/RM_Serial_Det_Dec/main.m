close all;
clear all;
clc;
%%
iter_num  = 80;
delta_f = 1.5e4;
Ns=4;               % Ns: number of transmit antennas
Nr=4;               % Nr:number of received antennas
M = 32; 
N = 8; 
MN = M*N;
N_x=Ns*MN;
N_y=Nr*MN;
dop = 100*(1e3/3600)*(4e9/3e8);% maximum Doppler, 450km/h as an example

Doppler_taps_max = round(dop*N/delta_f);
Epsilon=N; % 0 for integer Doppler shift and >0 for fractional Doppler 
c1=(2*(Doppler_taps_max+Epsilon)+1)/(2*MN); % c1 equation (48)
c2=1e-5; % c2 should be much smaller than 1/(2NN)

aa=1:N_x;
aa=aa.';

Phi_c1=exp(-1j*2*pi*c1*(aa-1).^2);
Phi_c2=exp(-1j*2*pi*c2*(aa-1).^2);

fs_N = 1;
fs = fs_N*M*delta_f;
beta = 0.4;
SNR = [0:1:10];
Num_MC = 5000;
thre=0;
InfoLen=1204;
CodeRate =  602/1024 ;  
cbsInfo = nrULSCHInfo(InfoLen,CodeRate);
A20=kron(dftmtx(N)/sqrt(N),eye(M)); % OTFS
A2=kron(eye(Ns),A20);
A1=diag(Phi_c2)*dftmtx(N_x)/sqrt(N_x)*diag(Phi_c1); % AFDM
A4=dftmtx(N_x)/sqrt(N_x); % OFDM
rho=0.5;
ber_mamp3=zeros(Num_MC,length(SNR));
ber_oamp1=zeros(Num_MC,length(SNR));
ber_oamp2=zeros(Num_MC,length(SNR));
ber_oamp4=zeros(Num_MC,length(SNR));
bler_mamp3=zeros(1,length(SNR));
bler_oamp2=zeros(1,length(SNR));
bler_oamp1=zeros(1,length(SNR));
bler_oamp4=zeros(1,length(SNR));
for j = 1:Num_MC
    j
    orderm=randperm(N_x,N_x);
    Em=eye(N_x);
    Sm=Em(orderm,:);
    A3=dftmtx(N_x)/sqrt(N_x)*Sm; % RM
    %% Encode
    data = randi([0 1],InfoLen,1);
    % CRC 
    tbIn = nrCRCEncode(data,cbsInfo.CRC);
    % Code block segmentation
    cbsIn = nrCodeBlockSegmentLDPC(tbIn,cbsInfo.BGN);
    % LDPC 
    LDPC_enc = nrLDPCEncode(cbsIn,cbsInfo.BGN);
            
    RateMatchLength=ceil(InfoLen/CodeRate);
    RateMatch = nrRateMatchLDPC(LDPC_enc,RateMatchLength,0,"QPSK",1);

    % [xx,x] = Get_modulation(2,length(RateMatch)/2,RateMatch);
    [x] = nrSymbolModulate(RateMatch,'QPSK');
    

    %% Modulation
    s4_AFDM=A4'*x; %OFDM
    s3_AFDM=A3'*x; %RM
    s2_AFDM=A2'*x; %OTFS
    s1_AFDM=A1'*x; %AFDM

    %% channel model

    H_TD_ce=cell(Ns,Nr);
    P = 5;%Number of Path
    delayspread = 1e-7;
    index_D = 1;%1 means has Doppler, 0 means no Doppler 
    for k=1:P
       A(:,:,k)=relatedh(Nr,Ns,rho);
    end
    fs_N = 1;
    fs=fs_N*M*delta_f;
    H_DD_ce=cell(Ns,Nr);
    H_TD_ce=cell(Ns,Nr);
    for ns=1:Ns
       for nr=1:Nr
          [GG_m_0,g_m_0,Dopp_m_0,pdb_m_0,tau_m_0,NN_m_0,t_m_0] = getchannel_related(M,N,delta_f,fs,fs_N,zeros(MN,1),P,index_D,dop,beta,A(nr,ns,:),delayspread);
           H_DD_ce{nr,ns}=GG_m_0;
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
     H_TD_m=cell2mat(H_TD_ce);%  %Time domain channel 
     %MIMO channel normlization
    [~,H_TD_m]=Channel_norm(H_TD_m);
    [r4_m_AFDM] = H_TD_m*s4_AFDM;
    [r3_m_AFDM] = H_TD_m*s3_AFDM;
    [r2_m_AFDM] = H_TD_m*s2_AFDM;
    [r1_m_AFDM] = H_TD_m*s1_AFDM;
   
    % RM
    for i = 1:length(SNR)
        N0     = 1 * 10 .^ (SNR(i) * -0.1); 
        sigma  = (0.5 * N0) .^ 0.5;
        noise = sigma * (randn(N_y, 1) + 1i*randn(N_y, 1));
        Sn3_AFDM = r3_m_AFDM + noise;
        LL=3; 
        [mse_mamp3(j,i),ber_mamp3(j,i),numerr]=MAMP_CD_M(H_TD_m,Sn3_AFDM,N0,x,data,iter_num,A3,cbsInfo,InfoLen,CodeRate);
        if numerr~=0
            bler_mamp3(i)=bler_mamp3(i)+1;
        end
    end
    % OTFS
    for i = 1:length(SNR)  
        N0     = 1 * 10 .^ (SNR(i) * -0.1); 
        sigma  = (0.5 * N0) .^ 0.5;
        noise = sigma * (randn(N_y, 1) + 1i*randn(N_y, 1));
        Sn2_AFDM = r2_m_AFDM + noise;
        LL=3; 
        [mse_oamp2(j,i),ber_oamp2(j,i),numerr]=OAMP_det_CD_M(H_TD_m,Sn2_AFDM,N0,x,data,iter_num,A2,cbsInfo,InfoLen,CodeRate);
        if numerr~=0
            bler_oamp2(i)=bler_oamp2(i)+1;
        end 
    end
    % AFDM
    for i = 1:length(SNR)  
        N0     = 1 * 10 .^ (SNR(i) * -0.1); 
        sigma  = (0.5 * N0) .^ 0.5;
        noise = sigma * (randn(N_y, 1) + 1i*randn(N_y, 1));
        Sn1_AFDM = r1_m_AFDM + noise;
        LL=3; 
        [mse_oamp1(j,i),ber_oamp1(j,i),numerr]=OAMP_det_CD_M(H_TD_m,Sn1_AFDM,N0,x,data,iter_num,A1,cbsInfo,InfoLen,CodeRate);
        if numerr~=0
            bler_oamp1(i)=bler_oamp1(i)+1;
        end  
    end
    % OFDM
    for i = 1:length(SNR)  
        N0     = 1 * 10 .^ (SNR(i) * -0.1); 
        sigma  = (0.5 * N0) .^ 0.5;
        noise = sigma * (randn(N_y, 1) + 1i*randn(N_y, 1));
        Sn4_AFDM = r4_m_AFDM + noise;
        LL=3; 
        [mse_oamp4(j,i),ber_oamp4(j,i),numerr]=OAMP_det_CD_M(H_TD_m,Sn4_AFDM,N0,x,data,iter_num,A4,cbsInfo,InfoLen,CodeRate);
        if numerr~=0
            bler_oamp4(i)=bler_oamp4(i)+1;
        end 
    end
end
%%
mbler_mamp3=bler_mamp3/Num_MC;
mbler_oamp2=bler_oamp2/Num_MC;
mbler_oamp1=bler_oamp1/Num_MC;
mbler_oamp4=bler_oamp4/Num_MC;
mBER_RM=mean(ber_mamp3);
oBER_OTFS=mean(ber_oamp2);
oBER_AFDM=mean(ber_oamp1);
oBER_OFDM=mean(ber_oamp4);
%%
figure;
semilogy(SNR,mBER_RM, 'LineWidth', 1.2);
hold on;
semilogy(SNR,oBER_OTFS, 'LineWidth', 1.2);
hold on;
semilogy(SNR,oBER_AFDM, 'LineWidth', 1.2);
hold on;
semilogy(SNR,oBER_OFDM, 'LineWidth', 1.2);
% xlim([4 9]);
% ylim([1e-5 5e-1])
legend('RM-CD-MAMP','OTFS-CD-OAMP','AFDM-CD-OAMP','OFDM-CD-OAMP')
title(['BER-QPSK ','M=',num2str(M),',N=',num2str(N),',R=',num2str(CodeRate),',InfoLen=',num2str(InfoLen),' Ns=',num2str(Ns),',Nr=',num2str(Nr)]);
xlabel('SNR(dB)', 'FontSize', 11);
%%
figure;
semilogy(SNR,mbler_mamp3, 'LineWidth', 1.2);
hold on;
semilogy(SNR,mbler_oamp2, 'LineWidth', 1.2);
hold on;
semilogy(SNR,mbler_oamp1, 'LineWidth', 1.2);
hold on;
semilogy(SNR,mbler_oamp4, 'LineWidth', 1.2);
% xlim([4 9]);
legend('RM-CD-MAMP','OTFS-CD-OAMP','AFDM-CD-OAMP','OFDM-CD-OAMP')
title(['BLER-QPSK ','M=',num2str(M),',N=',num2str(N),'R=',num2str(CodeRate),',InfoLen=',num2str(InfoLen),' Ns=',num2str(Ns),',Nr=',num2str(Nr)]);
xlabel('SNR(dB)', 'FontSize', 11);