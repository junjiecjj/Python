function [AUD_Index_K_set, H_set_hat] = A21_BOMP_AUD_CE(Y_set, Z_set, K, P_subc_sel_set, epsilon)

    [GM,K_Nbs] = size(Z_set);
    N_BS = K_Nbs/K;
    
    %% Iterations
    AUD_Index_K_set = [];
    H_set_hat = zeros(N_BS,numel(P_subc_sel_set),K);
    for pp = 1:numel(P_subc_sel_set)
        % Initializations
        iter_num = 0;
        AUD_Index_set_p = [];	% support set for p-th subc
        Z_active_set_p = [];
        Res_p = Y_set(:,P_subc_sel_set(pp));
        MSE_p = 2*epsilon;    % Pre-define MSE
        while (MSE_p > epsilon)
            % Distributed Correlation
            Corr_set_p = Z_set'*Res_p;

            % Find the maximum projection along the different spaces
            Corr_vec_p = zeros(K,1);
            for kk_1 = 1:K
                Corr_vec_p(kk_1) = sum(abs(Corr_set_p((kk_1-1)*N_BS+1:kk_1*N_BS)));
            end
            [~, index_ka_p] = max(Corr_vec_p);

            % Update the current guess of the common support
%             AUD_Index_set = union(AUD_Index_set, index_ka); % union:ascending sort (default)
            AUD_Index_set_p = [AUD_Index_set_p, index_ka_p];

            % Project the input signal onto the subspace given by the support using LS
            Z_active_set_p = [Z_active_set_p, Z_set(:,(index_ka_p-1)*N_BS+1:index_ka_p*N_BS)];
            h_AUD_hat_p = pinv(Z_active_set_p)*Y_set(:,P_subc_sel_set(pp));

            % Update residual
            Res_p = Y_set(:,P_subc_sel_set(pp)) - Z_active_set_p*h_AUD_hat_p;
            
            % Compute the current MSE
            MSE_p = trace(Res_p'*Res_p)/GM;
            
            % Compte the number of iteration
            iter_num = iter_num + 1;
        end
        AUD_Index_K_set = union(AUD_Index_K_set, AUD_Index_set_p);
        
        % Assign estimated complex channel gains
        for kk_a = 1:numel(AUD_Index_set_p)
            H_set_hat(:,pp,AUD_Index_set_p(kk_a)) = h_AUD_hat_p((kk_a-1)*N_BS+1:kk_a*N_BS);
        end
    end
    
end
