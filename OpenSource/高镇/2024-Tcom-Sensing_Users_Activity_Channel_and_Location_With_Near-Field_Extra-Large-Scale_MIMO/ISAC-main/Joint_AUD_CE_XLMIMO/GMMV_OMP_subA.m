function [AUD_Index_K_set, H_set_hat] = GMMV_OMP_subA(Y_set, Z_set, K, epsilon, M_suba)

% 根据子阵列作支撑集和LS估计

[GM,P_subc] = size(Y_set);
KNbs = size(Z_set,2);
N_BS = KNbs/K;
N_sub = N_BS/M_suba;

% Initialize the residual vectors to the input signal vectors and support estimate
AUD_Index_set = [];
Z_Col_Index_set = [];
Res_set = Y_set;
MSE = 2*epsilon;    % Pre-define MSE
iter_num = 1;       % Initialize the number of iteration

%% Find AUD support and estimate channels
NorPre=1000;
while (1)
    % Distributed Correlation
    Corr_set = sum(abs(reshape(pagemtimes(reshape(Res_set,1,GM,P_subc), conj(Z_set)), KNbs,P_subc)) ,2);
    % Find the maximum projection along the different spaces
    Corr_vec = sum(reshape(Corr_set,N_sub,K*M_suba));
    % Corr_vec2 = reshape(sum(reshape(Corr_set, N_sub, K*M_suba)),M_suba,K);
    [~, index_ka] = max(Corr_vec);
    % [~, index_ka_subA] = max(Corr_vec2(:,index_ka));
    % Indd=[Indd,[index_ka;index_ka_subA]];
    % Update the current guess of the common support
    AUD_Index_set = union(AUD_Index_set, index_ka);

    % % Distributed Correlation
    % Corr_set = zeros(KNbs,1);
    % for pp_1 = 1:P_subc
    %     Corr_set = Corr_set + abs(Z_set(:,:,pp_1)'*Res_set(:,pp_1));
    % end
    % 
    % % Find the maximum projection along the different spaces
    % Corr_vec = sum(reshape(Corr_set,N_sub,K*M_suba));
    % [~, index_ka] = max(Corr_vec);
    % 
    % % Update the current guess of the common support
    % AUD_Index_set = [AUD_Index_set, index_ka];
    Z_Col_Index_iter = (index_ka-1)*N_sub+1:index_ka*N_sub;
    Z_Col_Index_set = [Z_Col_Index_set, Z_Col_Index_iter]; % block index set
    
    h_p_hat_set = zeros(size(AUD_Index_set,2)*N_sub,P_subc);
    for pp_2 = 1:P_subc
        Z_Col_Index_p = Z_set(:,Z_Col_Index_set,pp_2);

        % Project the input signal onto the subspace given by the support using LS
        h_p_hat_set(:,pp_2) = pinv(Z_Col_Index_p)*Y_set(:,pp_2);

        % Update residual
        Res_set(:,pp_2) = Y_set(:,pp_2) - Z_Col_Index_p*h_p_hat_set(:,pp_2);
    end
    % Z_p = Z_set(:,Z_Col_Index_set,:);
    % TMP = conj(permute(Z_p,[2,1,3]));
    % h00 = pagemtimes(pagemtimes(pageinv(pagemtimes(TMP, Z_p)),TMP), reshape(Y_set,GM,1,P_subc));
    % h_p_hat_set = reshape(h00, size(h00,1),P_subc);
    % Res_set = Y_set - reshape(pagemtimes(Z_p, h00), GM, P_subc);

    % Compute the current MSE
    MSE = trace(Res_set'*Res_set)/(GM*P_subc);
    
    

    % Compute the number of iteration
    iter_num = iter_num + 1;
    % TQQP = ceil(AUD_Index_set/M_suba);
    % Index_set = unique(TQQP);
    if MSE <= epsilon
        break;
    end
    NorNew = norm(Res_set,'fro');
    if NorNew >= NorPre
        break;
    end
    NorPre = NorNew;
end
AUD_Index_K_set = unique(ceil(AUD_Index_set/M_suba));

%% Assign estimated complex channel gains
H_set_hat = zeros(N_BS*K,P_subc);
H_set_hat(Z_Col_Index_set,:)=h_p_hat_set;
H_set_hat = reshape(H_set_hat, [N_BS,K,P_subc]);
H_set_hat = permute(H_set_hat,[1,3,2]);

end