%% Parameter Initialization

function achievableRateErgodic(Num_MC,Nr,Ns,N,M,rhot,rhor,P,max_speed)

%% Monte Carlo SE 

fs_N=1;
delta_f=15000;
delta =1;
fs=fs_N*M*delta_f;
MN=M*N;


index_D = 1;            %1 means has Doppler, 0 means no Doppler
car_fre = 4*10^9;      % Carrier frequency

dop = max_speed*(1e3/3600)*(car_fre/3e8);      % maximum Doppler, 150km/h as an example
beta=0.4;
if Ns>1
    snr_dB =[-20:0.5:20];
else
    snr_dB =[-10:0.5:20];
end
% snr_dB =[2];
snr=10.^(snr_dB/10);
v_LE    =[1e-5:1e-4:0.1, 0.1:0.01:1];% 
RHO_LE_OTFS=zeros(length(snr_dB),length(v_LE));%rho的线
%% OTFS param
A2=kron(dftmtx(N)/sqrt(N),eye(M));%OTFS
A2_nr=kron(eye(Nr),A2);
A2_ns=kron(eye(Ns),A2');

%%
for j=1:Num_MC
    j
    %% MIMO channel model
    A=zeros(Nr,Ns,P);
    for i=1:P
    A(:,:,i)=relatedh(Nr,Ns,rhot,rhor);
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
   %% 逐天线转为DD域信道
    H_DD_m_OTFS=A2_nr*H_TD_m*A2_ns;
    [~, Dia_OTFS, ~] = svd( H_DD_m_OTFS);
    dia_OTFS   = diag(Dia_OTFS);
    dia_OTFS   = dia_OTFS.';
    %%
     Sigma_n= snr.^-1;
    %%  SE
    vpre_LE=[1e-5:1e-4:0.1, 0.1:0.01:1];%线性端 不同输入均方误差 
    vpost_LE_OTFS= LE_OAMP_SE(vpre_LE, dia_OTFS, Sigma_n, Nr*MN, Ns*MN);

    %% OTFS
%     v_LE    =[1e-5:1e-4:0.1, 0.1:0.01:1];% 
    for vid = 1:length(snr_dB)
        v_mmse(vid,:) = GOAMP_GLE_inv_SE(v_LE, vpre_LE, vpost_LE_OTFS(vid,:));%
        rho_LE(vid,:) = 1./v_LE- 1./v_mmse(vid,:);%η检测端 对应rho计算 检测的线
        RHO_LE_OTFS(vid,:)= RHO_LE_OTFS(vid,:)+rho_LE(vid,:);
    end    
end
%% Achievable Rate of OAMP OTFS

RHO_LE_OTFS=RHO_LE_OTFS./Num_MC;
PHI_1_OTFS =RHO_LE_OTFS(:,end);
SNR_end_OTFS = RHO_LE_OTFS(:,1);

fprintf('OTFS Achievable Rate \n');
fprintf('OAMP Gaussian \n');
[R_OAMP_Gaussian_OTFS, R_Sep_Gaussian_OTFS]= Rate_OAMP_Gaussian(snr,v_LE, RHO_LE_OTFS, PHI_1_OTFS,SNR_end_OTFS);  %计算面积
fprintf('OAMP QPSK \n');
[R_OAMP_QPSK_OTFS, R_Sep_QPSK_OTFS] = Rate_OAMP_QPSK(snr,v_LE, RHO_LE_OTFS, PHI_1_OTFS,SNR_end_OTFS);
fprintf('OAMP 16QAM \n');
[R_OAMP_16QAM_OTFS,  R_Sep_16QAM_OTFS]= Rate_OAMP_16QAM(snr,v_LE, RHO_LE_OTFS, PHI_1_OTFS,SNR_end_OTFS);
% [~, R_Sep_16QAM_OTFS]= Rate_OAMP_16QAM(snr,v_LE, RHO_LE_OTFS, PHI_1_OTFS,SNR_end_OTFS);

% save(['sep16QAM','rhot=',num2str(rhot),'rhor=',num2str(rhor),'speed=',num2str(max_speed),'.mat'],'R_Sep_16QAM_OTFS','rhot','rhor','snr_dB')
save(['N=',num2str(N),'M=',num2str(M),'P=',num2str(P),'Ns=',num2str(Ns),'Nr=',num2str(Nr),'rho=',num2str(rhot),'speed=',num2str(max_speed),'.mat'])

end
