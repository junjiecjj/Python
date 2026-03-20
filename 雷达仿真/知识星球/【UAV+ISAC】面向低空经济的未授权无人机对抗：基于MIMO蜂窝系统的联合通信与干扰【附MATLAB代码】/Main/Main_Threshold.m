% Select the best eta to minimize the P_t
addpath("..\Plot\")
addpath("..\Function\")
if isempty(gcp('nocreate'))   
    numWorkers = 10;      
    parpool('local', numWorkers);
end
close all;
clc; clear
% run('D:\Code\Matlab\CVX\cvx\cvx_setup.m');
rng(999)
%% 1. Parameter Definition
R_th =      [5  7  9  7  7   7];
Gamma_th =  [13 13 13 3  -7  -17];
n_t = 16;
n_ue = 2;
n_uav = 2;
Nit = 2000;
Nit_each = ones(length(R_th),1)*Nit;
ErrorCase_Index = [];
ErrorIt_Index = [];
P_12 = zeros(length(R_th), Nit);
P_123 = zeros(length(R_th), Nit);
MaxErrorRate = zeros(length(R_th), Nit);
MaxErrorSINR = zeros(length(R_th), Nit);

r_th = 7;
gamma_th = 13;
Para = ParaClass_NUENUAV(n_t, n_ue, n_uav, r_th, gamma_th);
%% |     Constraint 1: Achievable Rate
H_tilde_1 = zeros(Para.N_ue,...
            Para.N_ue*Para.N_t,...
            Para.N_ue);
H_tilde_2 = zeros(1, ...
            Para.N_ue*Para.N_t,...
            Para.N_ue);
A_1 = zeros(Para.N_ue*Para.N_t,...
            Para.N_ue*Para.N_t, ...
            Para.N_ue);
%% |     Constraint 2: SINR
H_bar = zeros(Para.N_ue,...
            Para.N_ue*Para.N_t,...
            Para.N_uav);
A_2 = zeros(Para.N_ue*Para.N_t,...
            Para.N_ue*Para.N_t, ...
            Para.N_uav);

%% |     Constraint 3: Non Block Diagonized
eta = [0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
A_3 = zeros(Para.N_ue*Para.N_t,...
            Para.N_ue*Para.N_t, ...
            Para.N_ue*Para.N_t-1,...
            length(eta));
I_bar = zeros(Para.N_ue*Para.N_t,...
        Para.N_ue*Para.N_t,...
        Para.N_ue*Para.N_t-1);
for i = 1:Para.N_ue*Para.N_t-1
    I_bar(1:Para.N_ue*Para.N_t,...
        1:Para.N_ue*Para.N_t,...
        i) = [circshift(eye(Para.N_t*Para.N_ue), i)];
end
gamma_3 = 0;

% To use parfor, we need to predefine some parameters
N_t = Para.N_t;
N_ue = Para.N_ue;
N_uav = Para.N_uav;
sigma2_ue = Para.sigma2_ue;
sigma2_uav = Para.sigma2_uav;
P_e = Para.P_e;

for i_case = 1:length(R_th)
    r_th = R_th(i_case);
    gamma_th = Gamma_th(i_case);
    Para = ParaClass_NUENUAV(n_t, n_ue, n_uav, r_th, gamma_th);
    
    for i_it = 1:Nit    
        tic
        Para = Para.ParaUpd();
        disp(['i_case/N_case = ' num2str(i_case) '/' num2str(length(R_th))])
        disp(['r_th = ' num2str(r_th)])
        disp(['gamma_th = ' num2str(gamma_th)])
        disp(['i_it/Nit = ' num2str(i_it) '/' num2str(Nit)])
%% 2. Precoding Matrix Computation
%% |     Construct Standart SDP Form 
%% |     |      Constraint 1: Achievable Rate
        gamma_1 = (2^Para.R_th - 1)/2^Para.R_th*Para.sigma2_ue;
        for i_ue = 1:n_ue
            H_tilde_1(:,:,i_ue) = kron(eye(Para.N_ue),...
                        Para.H_ue(i_ue,:));
            H_tilde_2(1, 1+(i_ue-1)*Para.N_t:i_ue*Para.N_t, i_ue) = Para.H_ue(i_ue,:);
            A_1(:,:,i_ue) = H_tilde_2(:,:,i_ue)'*H_tilde_2(:,:,i_ue) - ...
                        (2^Para.R_th - 1)/2^Para.R_th*H_tilde_1(:,:,i_ue)'*H_tilde_1(:,:,i_ue);
        end    
        
%% |     |      Constraint 2: SINR
        gamma_2 = Para.P_e*10^(-Para.Gamma_th/10) - Para.sigma2_uav;
        for i_uav = 1:n_uav
            H_bar(:,:,i_uav) = kron(eye(Para.N_ue),...
                        Para.H_uav(i_uav,:));
            A_2(:,:,i_uav) = H_bar(:,:,i_uav)'*H_bar(:,:,i_uav);
        end
        
%% |     |      Constraint 3: Non Block Diagonized
        R = zeros(Para.N_ue, length(eta));
        Gamma = zeros(Para.N_uav, length(eta));
        P_cvx = zeros(1, length(eta));
        P_real = zeros(1, length(eta));
        F = zeros(Para.N_t, Para.N_ue, length(eta));
        
        for i_eta = 1:length(eta)    
            for i = 1:Para.N_ue-1
                A_3(:,:,i,i_eta) = I_bar(:,:,i) - eta(i_eta)*eye(Para.N_ue*Para.N_t);
            end
        end
        % To use parfor, we need to predefine some parameters
        H_ue = Para.H_ue;
        H_uav = Para.H_uav;
        parfor i_eta = 1:length(eta)    
        %     disp(['i_eta = ' num2str(i_eta)])
%% |     Compute SDP with Constraint 3
            % cvx_solver SDPT3
            F_tilde = ObtCvxSolu_NUENUAV(A_1, gamma_1, ...
                                 A_2, gamma_2, ...                            
                                 A_3(:,:,:,i_eta), gamma_3, ...
                                 N_t, N_ue, N_uav, 1);
            
            % % |     Obtain the Solution to the Original Problem
            
            try
                [V,D] = eigs(F_tilde);
            catch           
                continue
            end
            [value,num] = max(diag(D));
            f = sqrt(value)*V(:,num);
            F_hat = reshape(f,N_t,N_ue);    
%% |     Save Current Results
            for i_ue = 1:N_ue
                R(i_ue,i_eta) = real(log2(1 + H_ue(i_ue,:)*F_hat(:,i_ue)*F_hat(:,i_ue)'*H_ue(i_ue,:)'/...
                    (sigma2_ue + H_ue(i_ue,:)*(F_hat*F_hat')*H_ue(i_ue,:)'-...
                    H_ue(i_ue,:)*F_hat(:,i_ue)*F_hat(:,i_ue)'*H_ue(i_ue,:)')));
            end
            for i_uav = 1:N_uav
                Gamma(i_uav,i_eta) = real(10*log10(P_e/...
                    (H_uav(i_uav,:)*(F_hat*F_hat')*H_uav(i_uav,:)' + sigma2_uav)));
            end
            P_cvx(i_eta) = real(trace(F_tilde));
            P_real(i_eta) = real(trace(F_hat'*F_hat));
            F(:,:,i_eta) = F_hat;
        end
        if any(P_real == 0)
            ErrorCase_Index = [ErrorCase_Index i_case];
            ErrorIt_Index = [ErrorIt_Index i_it];
            Nit_each(i_case) = Nit_each(i_case) -1;
        else
%% |     Compute SDP without Constraint 3
            F_tilde_without3 = ObtCvxSolu_NUENUAV(A_1, gamma_1, ...
                                     A_2, gamma_2, ...                            
                                     A_3(:,:,:,1), gamma_3, ...
                                     N_t, N_ue, N_uav, 0);
            P_12(i_case, i_it) = real(trace(F_tilde_without3));
%% 3. Select the Best Precoding Matrix
            Error = zeros(1, length(eta));
            for i_eta = 1:length(eta)
                % mean error minimized
                %     Error(i_eta) = sum(abs(R(:,i_eta) - Para.R_th))+...
                %                    sum(abs(Gamma(:,i_eta) - Para.Gamma_th));
                % max error minimized
                Error(i_eta) = max(max(max(Para.R_th - R(:,i_eta)), 0),...
                               max(max(Gamma(:,i_eta) - Para.Gamma_th), 0));
            end     
            [~, idx_final] = min(Error);
            for i_eta = 1:length(eta)
                if Error(i_eta) < 1e-2
                    if  real(P_real(i_eta)) <  real(P_real(idx_final))
                        idx_final = i_eta;
                    end
                end
            end
            MaxErrorRate(i_case, i_it) = max(max(Para.R_th - R(:,idx_final)), 0);
            MaxErrorSINR(i_case, i_it) = max(max(Gamma(:,idx_final) - Para.Gamma_th), 0);            
%% 4. Save and Plot
            P_123(i_case, i_it) = real(P_real(idx_final));
            
            % Plot_Beampattern_NUENUAV(F(:,:,idx_final), Para);
            disp(['eta = ' num2str(eta(idx_final))])
            disp(['P_12 = ' num2str(P_12(i_case, i_it))])
            disp(['trace(F_tilde) = ' num2str(P_cvx(idx_final))])
            disp(['trace(F_hat^H*F_hat) = ' num2str(P_real(idx_final))])
            disp('-------------------------------------------')
            for i_ue = 1:Para.N_ue
                disp(['R_' num2str(i_ue) ' = ' num2str(real(R(i_ue,idx_final)))])
            end
            disp(['R_th = ' num2str(Para.R_th)])
            disp('-------------------------------------------')
            for i_uav = 1:Para.N_uav
                disp(['Gamma_' num2str(i_uav) ' = ' num2str(real(Gamma(i_uav,idx_final)))])
            end
            disp(['Gamma_th = ' num2str(Para.Gamma_th)]) 
            disp('-------------------------------------------')
            disp(['MaxErrorRate = ' num2str(MaxErrorRate(i_case, i_it))])
            disp(['MaxErrorSINR = ' num2str(MaxErrorSINR(i_case, i_it))])            
        end
        toc
        disp('============================================================')
    end
    save('..\Data\Threshold_it_2000.mat', ...
            'R_th', ...
            'Gamma_th', ...
            'n_ue', ...
            'n_uav', ...
            'Nit', ...
            'Nit_each',...
            'ErrorCase_Index',...
            'ErrorIt_Index',...
            'P_12', ...
            'P_123',...
            'MaxErrorRate',...
            'MaxErrorSINR')
end