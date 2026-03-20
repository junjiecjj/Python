clear all;
clc;
% close all;
addpath('common\');
addpath('limits\');
%%
% system parameters
% Num_MC=100;
Num_MC=20;
fs_N=1;
delta_f=15000;
P = 5;                       %Number of Path
index_D = 1;            %1 means has Doppler, 0 means no Doppler
car_fre = 4*10^9;      % Carrier frequency
max_speed=500;     %150km/h
dop = max_speed*(1e3/3600)*(car_fre/3e8);      % maximum Doppler, 150km/h as an example
beta=0.4;
snr_dB =1;%-11.1
snr=10.^(snr_dB/10);
v_LE    =[1e-5:1e-4:0.1, 0.1:0.01:1];% 
RHO_LE=zeros(length(snr_dB),length(v_LE));%rho
for j=1:Num_MC
    j
    %% MIMO channel model
    Nr=1;
    Ns=1;
    N =8;
    delta =1;
    M = 32;
    rho=0.2;
    MN=M*N;
    fs=fs_N*M*delta_f;
    
    A=zeros(Nr,Ns,P);
    for i=1:P
    A(:,:,i)=relatedh(Nr,Ns,rho);
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
    H_TD_m=cell2mat(H_TD_ce);
    [~,H_TD_m]=channel_norm(H_TD_m);
    [U, Dia, V] = svd( H_TD_m);
    dia   = diag(Dia);
    dia=dia.';
    %%
    % system parametersdd
     Sigma_n= snr.^-1;
    %%  SE
    vpre_LE=[1e-5:1e-4:0.1, 0.1:0.01:1];%线性端 不同输入均方误差 
    % %vpost_LE=GOAMP_GLE_Clip_SE(delta, clip, snr.^-1, vpre_LE);

    vpost_LE= LE_OAMP_SE(vpre_LE, dia, Sigma_n, Nr*MN, Ns*MN);%利用OAMP的方差输入输出关系有解析表达式，计算后验方差，不同信噪比下
%     v_LE    =[1e-5:1e-4:0.1, 0.1:0.01:1];% 
    for vid = 1:length(snr_dB)
        v_mmse(vid,:) = GOAMP_GLE_inv_SE(v_LE, vpre_LE, vpost_LE(vid,:));%
        rho_LE(vid,:) = 1./v_LE- 1./v_mmse(vid,:);%η检测端 对应rho计算 检测的线
        RHO_LE(vid,:)= RHO_LE(vid,:)+rho_LE(vid,:);
    end
   
    % all matries in monte carlo
    
end
%%
RHO_LE=RHO_LE./Num_MC;
PHI_1 =RHO_LE(:,end)';
SNR_end = RHO_LE(:,1)';

%% code parameters
feedback = 'P';
Thres    = 0.001;
truncated_v = 1e-3
CodeLen = 102400;%100000
MaxTrial = 20;
deg_set  = [2:1:30, 30:5:50, 60:10:140, 150:50:500, 600:100:1000, 1500:500:5000, 6000:1000:10000, 20000:10000:100000];  
deg_set  = deg_set(deg_set<=5000);
dv_init = [2];
b_init  = [1];
% dc_init = [8, 30];%[6 25]
% d_init  = [0.8, 0.2];%[0.4 0.6]
dc_init = [6];%[6 25]
d_init  = [1];%[0.4 0.6]

b_ik = 100; 
user =0;  %0: symmetric, 1: aysmmetirc user1, 2:asymmetirc user2
%%
%SE transfer
figure;
[var, rho] = Clip_OAMP_SEtransfer(SNR_end,v_LE, RHO_LE, PHI_1);

plot(rho, var, '-r');

% [ var, rho] = OAMP_SEtransfer(M,K, beta, kappa, snr);
% plot(rho,var,'r');
%[vstar1, rho1, vstar2, rho2, var, rho, v_star,rho_star,v_s, C] = OAMP_SEtransfer(K, beta, kappa, snr);
%% set output path
OutputPath = sprintf('user%g_delta%g_b%g_dv%d_dc%d_snrdB=%g_N%d', user,delta, b_ik, max(deg_set),max(dc_init), snr_dB, CodeLen);
if(~exist(OutputPath,'dir')) 
    mkdir(OutputPath);
end
%% asymmetric user SNR rho
% [vk1,vk2, v1_part2, v2_part2] = AsymRate(var,rho,b_ik,vstar1,rho1, vstar2, rho2, v_star,v_s);
% save([OutputPath '/Output_SE.mat'], 'rho', 'var', 'rho2', 'v1_part2', 'v2_part2');
%% optimize LDPC codes
% symmetric
if (user ==0)
    [variable_edge_degree, variable_edge_weight, parity_edge_degree, parity_edge_weight, snr_dec_ap, ~, v_dec_post, ~] =...
        LDPC_optimization_function(truncated_v, feedback, Thres, OutputPath, MaxTrial, deg_set, rho,...
        var, dv_init, b_init, dc_init, d_init, 4);
        %produce LDPC parity-check matrix
    if(CodeLen<=0)
        display('CodeLen <= 0, do not produce LDPC code...');
    else
        display('degree is ready, producing LDPC code...');
        LDPC_produce(variable_edge_degree, variable_edge_weight,parity_edge_degree, parity_edge_weight, CodeLen, OutputPath);
        display(['LDPC parity-check matrix is produced in the folder "' OutputPath '".']);
        save([OutputPath '/LDPC_OAMP_main.mat']);
        toc;
        
        %revisit transfer functions of the output HFile evolution
        [dv, b, dc, d] = LDPC_check_distribution([OutputPath '/H_Fine.txt']);
        save([OutputPath '/Output_dist.mat'], 'dv', 'b', 'dc', 'd', 'snr_dB', 'delta');
        [snr_dec_HFine, ~, v_dec_HFine, ~] = LDPC_transfer(dv,b,dc,d,rho,var,feedback,-1);
        %%
        figure();
        subplot(2,1,1);
        plot(rho, var, 'r-',...
            snr_dec_ap, v_dec_post, 'g-',...
            snr_dec_HFine, v_dec_HFine, 'm--');
        hold on;
        xlabel('\rho');
        ylabel('v');
        legend('DET', 'DEC approx, opt', 'DEC approx, HFine', 'location', 'best');
        title('transfer revisit linear');
        subplot(2,1,2);
        semilogy(10*log10(rho), var, 'r-',...
            10*log10(snr_dec_ap), v_dec_post, 'g-',...
            10*log10(snr_dec_HFine), v_dec_HFine, 'm--');
        hold on;
        xlabel('\rho');
        ylabel('v');
        legend('DET', 'DEC approx, opt', 'DEC approx, HFine', 'location', 'best');
        title('transfer revisit log');
        %%
        saveas(gcf,[OutputPath '/5_transfer_HFine.fig']);
        saveas(gcf,[OutputPath '/5_transfer_HFine.png']);
        save([OutputPath '/LDPC_OAMP_main.mat']);
    end%end CodeLen<=0.0
    
end

%user1
if (user ==1)
    [variable_edge_degree1, variable_edge_weight1, parity_edge_degree1, parity_edge_weight1, snr_dec_ap1, ~, v_dec_post1, ~] =...
        LDPC_optimization_function(truncated_v, feedback, Thres, OutputPath, MaxTrial, deg_set, rho,...
        vk1, dv_init, b_init, dc_init, d_init, 2);
    
    %produce LDPC parity-check matrix
    if(CodeLen<=0)
        display('CodeLen <= 0, do not produce LDPC code...');
    else
        display('degree is ready, producing LDPC code...');
        LDPC_produce(variable_edge_degree1, variable_edge_weight1,...
            parity_edge_degree1, parity_edge_weight1, CodeLen, OutputPath);
        display(['LDPC parity-check matrix is produced in the folder "' OutputPath '".']);
        save([OutputPath '/LDPC_OAMP_main1.mat']);
        toc;
        
        %revisit transfer functions of the output HFile evolution
        [dv, b, dc, d] = LDPC_check_distribution([OutputPath '/H_Fine.txt']);
        save([OutputPath '/Output_dist.mat'], 'dv', 'b', 'dc', 'd', 'snr_dB', 'beta');
        [snr_dec_HFine, ~, v_dec_HFine, ~] = LDPC_transfer(dv,b,dc,d,rho,vk1,feedback,-1);
        figure(6);
        subplot(2,1,1);
        plot(rho, vk1, 'r-',...
            snr_dec_ap1, v_dec_post1, 'g-',...
            snr_dec_HFine, v_dec_HFine, 'm--');
        hold on;
        xlabel('\rho');
        ylabel('v');
        legend('DET', 'DEC approx, opt', 'DEC approx, HFine', 'location', 'best');
        title('transfer revisit linear');
        subplot(2,1,2);
        semilogy(10*log10(rho), vk1, 'r-',...
            10*log10(snr_dec_ap1), v_dec_post1, 'g-',...
            10*log10(snr_dec_HFine), v_dec_HFine, 'm--');
        hold on;
        xlabel('\rho');
        ylabel('v');
        legend('DET', 'DEC approx, opt', 'DEC approx, HFine', 'location', 'best');
        title('transfer revisit log');
        saveas(gcf,[OutputPath '/6_transfer_HFine.fig']);
        saveas(gcf,[OutputPath '/6_transfer_HFine.png']);
        save([OutputPath '/LDPC_OAMP_main.mat']);
    end%end CodeLen<=0.0
end
%% user2
if (user ==2)
    [variable_edge_degree2, variable_edge_weight2, parity_edge_degree2, parity_edge_weight2, snr_dec_ap2, ~, v_dec_post2, ~] =...
        LDPC_optimization_function(truncated_v, feedback, Thres, OutputPath, MaxTrial, deg_set, rho,...
        vk2, dv_init, b_init, dc_init, d_init, 3);
        %produce LDPC parity-check matrix
    if(CodeLen<=0)
        display('CodeLen <= 0, do not produce LDPC code...');
    else
        display('degree is ready, producing LDPC code...');
        LDPC_produce(variable_edge_degree2, variable_edge_weight2,...
            parity_edge_degree2, parity_edge_weight2, CodeLen, OutputPath);
        display(['LDPC parity-check matrix is produced in the folder "' OutputPath '".']);
        save([OutputPath '/LDPC_OAMP_main.mat']);
        toc;
        
        %revisit transfer functions of the output HFile evolution
        [dv, b, dc, d] = LDPC_check_distribution([OutputPath '/H_Fine.txt']);
        save([OutputPath '/Output_dist.mat'], 'dv', 'b', 'dc', 'd', 'snr_dB', 'beta');
        [snr_dec_HFine, ~, v_dec_HFine, ~] = LDPC_transfer(dv,b,dc,d,rho,vk2,feedback,-1);
        figure(7);
        subplot(2,1,1);
        plot(rho, vk2, 'r-',...
            snr_dec_ap2, v_dec_post2, 'g-',...
            snr_dec_HFine, v_dec_HFine, 'm--');
        hold on;
        xlabel('\rho');
        ylabel('v');
        legend('DET', 'DEC approx, opt', 'DEC approx, HFine', 'location', 'best');
        title('transfer revisit linear');
        subplot(2,1,2);
        semilogy(10*log10(rho), vk2, 'r-',...
            10*log10(snr_dec_ap2), v_dec_post2, 'g-',...
            10*log10(snr_dec_HFine), v_dec_HFine, 'm--');
        hold on;
        xlabel('\rho');
        ylabel('v');
        legend('DET', 'DEC approx, opt', 'DEC approx, HFine', 'location', 'best');
        title('transfer revisit log');
        saveas(gcf,[OutputPath '/7_transfer_HFine.fig']);
        saveas(gcf,[OutputPath '/7_transfer_HFine.png']);
        save([OutputPath '/LDPC_OAMP_main.mat']);
    end%end CodeLen<=0.0
end
function v = MMSE_QPSK(snr)
%snr = snr/2;   % scale from BPSK for QPSK
max=100;
v = zeros(size(snr));
for i=1:length(snr)
    v(i) = 1 - integral(@(x) f_QPSK(x,snr(i)), -max, max);
end

%% 
function y = f_QPSK(x,snr) 
y  = exp(-x.^2/2)/sqrt(2*pi) .* tanh(snr - sqrt(snr).*x);
end
end


 function  v_post= LE_OAMP_SE(v, dia, v_nn, M, N)
 if N>M
    dia=dia.';
 end
v_post=zeros(length(v_nn),size(v,2));
for jj=1:length(v_nn)
    v_n=v_nn(jj);
    for i=1:length(v)
    rho = v_n / v(i);
    Dia = [dia.^2; zeros(N-M, 1)];
    D = 1 ./ (rho + Dia);
    v_post(jj,i) = v_n / N * sum(D);
    end  
end
 end