function [AUD_Index_K_set, H_set_hat] = MMV_OMP_A(Y_set, Z_set, K, epsilon, M_suba)

[GM,P_subc] = size(Y_set);
N_BS = size(Z_set,2)/K;
N_sub = N_BS/M_suba;
% Initialize the residual vectors to the input signal vectors and support estimate
AUD_Index_set = [];
Z_Col_Index_set = [];
Res_set = Y_set;
MSE = 2*epsilon;    % Pre-define MSE
iter_num = 1;       % Initialize the number of iteration

%% Find AUD support and estimate channels
NorPre=1000;
while (MSE > epsilon)
    % Distributed Correlation
    Corr_set = sum(abs(Z_set'*Res_set),2);
    
    % Find the maximum projection along the different spaces
    Corr_vec = sum(reshape(Corr_set,N_BS,K));
    [~, index_ka] = max(Corr_vec);
    
    % Update the current guess of the common support
    AUD_Index_set = union(AUD_Index_set, index_ka);
    
    % Project the input signal onto the subspace given by the support using LS
    Z_Col_Index_iter = (index_ka-1)*N_BS+1:index_ka*N_BS;
    Z_Col_Index_set = [Z_Col_Index_set, Z_Col_Index_iter];

    Z_active_set = Z_set(:,Z_Col_Index_set);
    H_AUD_hat_set = pinv(Z_active_set)*Y_set;
    
    % Update residual
    Res_set = Y_set - Z_active_set*H_AUD_hat_set;
    NorNew = norm(Res_set,'fro');
    if NorNew >=NorPre
        break;
    end
    NorPre = NorNew;
    % Compute the current MSE
    MSE = trace(Res_set'*Res_set)/(GM*P_subc);
    
    % Compte the number of iteration
    iter_num = iter_num + 1;
end
AUD_Index_K_set = sort(AUD_Index_set);

%% Assign estimated complex channel gains
H_set_hat0 = zeros(N_BS*K,P_subc);
H_set_hat0(Z_Col_Index_set, :) = H_AUD_hat_set;
H_set_hat = reshape(H_set_hat0, N_BS, K, P_subc);
H_set_hat = permute(H_set_hat, [1, 3, 2]);
end