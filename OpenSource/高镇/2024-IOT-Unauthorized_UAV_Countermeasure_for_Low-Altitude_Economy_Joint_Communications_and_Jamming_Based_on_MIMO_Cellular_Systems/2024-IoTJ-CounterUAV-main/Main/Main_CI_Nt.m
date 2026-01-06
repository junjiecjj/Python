% Select the best eta to minimize the P_t
addpath("..\Plot\")
addpath("..\Function\")
if isempty(gcp('nocreate'))   
    numWorkers = 10;      
    parpool('local', numWorkers);
end
% close all;
clc; clear
% run('D:\Code\Matlab\CVX\cvx\cvx_setup.m');
rng(999)
%% 1. Parameter Definition
N_t_all = [8 16 32 64];
n_t = 16;
n_ue = 2;
n_uav = 2;
Nit = 5000;
ErrorCase_Index = [];
ErrorIt_Index = [];
P = zeros(length(N_t_all), Nit);
% In order to let R_th =  7.33,
% we need to set R_th as follows
r_th = 7;
gamma_th = 13;
Para = ParaClass_NUENUAV(n_t, n_ue, n_uav, r_th, gamma_th);

% To use parfor, we need to predefine some parameters
sigma2_ue = Para.sigma2_ue;
sigma2_uav = Para.sigma2_uav;
P_e = Para.P_e;
%% 2. Generate Channel
parfor i_case = 1:length(N_t_all)
    n_t = N_t_all(i_case);
    Para = ParaClass_NUENUAV(n_t, n_ue, n_uav, r_th, gamma_th);

    for i_it = 1:Nit
        tic
        Para = Para.ParaUpd();
        R = zeros(Para.N_ue,1);
        Gamma = zeros(Para.N_uav,1);
        disp(['i_case/N_case = ' num2str(i_case) '/' num2str(length(N_t_all))])
        disp(['n_t = ' num2str(n_t)])
        disp(['n_ue = ' num2str(n_ue)])
        disp(['n_uav = ' num2str(n_uav)])
        disp(['i_it/Nit = ' num2str(i_it) '/' num2str(Nit)])
%% 3. Pseudo-Inverse
        F_norm = pinv(Para.H);
%% 4. Power Allocation
        F_hat = [sqrt((2^r_th-1)*sigma2_ue)*F_norm(:,1:Para.N_ue), ...
                 sqrt(10^(-gamma_th/10)*P_e - sigma2_uav)*F_norm(:,Para.N_ue+1:end)];
        P(i_case, i_it) = norm(F_hat,"fro")^2;

%% 5. Compute Information
        for i_ue = 1:Para.N_ue
            R(i_ue) = log2(1 + Para.H_ue(i_ue,:)*F_hat(:,i_ue)*F_hat(:,i_ue)'*Para.H_ue(i_ue,:)'/...
                (sigma2_ue + Para.H_ue(i_ue,:)*(F_hat*F_hat')*Para.H_ue(i_ue,:)'-...
                Para.H_ue(i_ue,:)*F_hat(:,i_ue)*F_hat(:,i_ue)'*Para.H_ue(i_ue,:)'));
        end
        for i_uav = 1:Para.N_uav
            Gamma(i_uav) = 10*log10(P_e/...
                (Para.H_uav(i_uav,:)*(F_hat*F_hat')*Para.H_uav(i_uav,:)' + sigma2_uav));
        end
        disp(['P = ' num2str(P(i_case, i_it))])
        disp('-------------------------------------------')
        for i_ue = 1:Para.N_ue
            disp(['R_' num2str(i_ue) ' = ' num2str(real(R(i_ue)))])
        end
        disp(['R_th = ' num2str(Para.R_th)])
        disp('-------------------------------------------')
        for i_uav = 1:Para.N_uav
            disp(['Gamma_' num2str(i_uav) ' = ' num2str(real(Gamma(i_uav)))])
        end
        disp(['Gamma_th = ' num2str(Para.Gamma_th)])   
        toc
        disp('============================================================')
    end       
end
%% 5. Save and Plot
save('..\Data\CI_Nt_it_5000.mat', ...
            'r_th', ...
            'gamma_th', ...
            'n_ue', ...
            'n_uav', ...
            'N_t_all',...
            'Nit', ...            
            'P')