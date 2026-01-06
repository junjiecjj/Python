function [AUD_Index_K_set, H_set_hat] = A22_BSP_AUD_CE(Y_set, F_set, K, Ka)

    [GM, P_subc] = size(Y_set);
    K_Nbs = size(F_set,2);
    N_BS = K_Nbs/K;

    %% Initializations
    AUD_Index_set = [];	% support set
    AUD_Index_set_iter = [];
    Res_set = Y_set;	% residual signals
    iter_max = Ka;
    
    %% Iterations
    for iter = 1:iter_max
        % Identification
        Res_set_pre = Res_set;
        Corr_set = F_set'*Res_set;
        
        Corr_vec = zeros(K,1);
        for kk_1 = 1:K
            Corr_vec(kk_1) = sum(sum(abs(Corr_set((kk_1-1)*N_BS+1:kk_1*N_BS,:))));
        end
        
        [~, index_up_init] = sort(Corr_vec, 'descend');
        index_up = index_up_init(1:Ka);
        
        % Support Merger
        AUD_Index_set = union(AUD_Index_set, index_up);	% union:ascending sort (default)
        
        % Least Square Estimation
        Ka_est = length(AUD_Index_set);
        F_set_temp1 = zeros(GM,Ka_est*N_BS);
        for kk_2 = 1:Ka_est
            Sel_Index_kk1 = AUD_Index_set(kk_2);
            F_set_temp1(:,(kk_2-1)*N_BS+1:kk_2*N_BS) = F_set(:,(Sel_Index_kk1-1)*N_BS+1:Sel_Index_kk1*N_BS);
        end
        H_ls_temp = pinv(F_set_temp1)*Y_set;
        
        % Support Pruning
        Prun_vec = zeros(Ka_est,1);
        for kk_3 = 1:Ka_est
            Prun_vec(kk_3) = sum(sum(abs(H_ls_temp((kk_3-1)*N_BS+1:kk_3*N_BS,:))));
        end
        [~, index_pru] = sort(Prun_vec, 'descend');
        
        % Final Support Update
        AUD_Index_set = AUD_Index_set(index_pru(1:Ka));
        F_set_temp_2 = zeros(GM,Ka*N_BS);
        H_ls = zeros(Ka*N_BS,P_subc);
        for kk_4 = 1:Ka
            Sel_Index_kk2 = index_pru(kk_4);
            F_set_temp_2(:,(kk_4-1)*N_BS+1:kk_4*N_BS) = F_set_temp1(:,(Sel_Index_kk2-1)*N_BS+1:Sel_Index_kk2*N_BS);
            H_ls((kk_4-1)*N_BS+1:kk_4*N_BS,:) = H_ls_temp((Sel_Index_kk2-1)*N_BS+1:Sel_Index_kk2*N_BS,:);
        end
%         AUD_Index_set = AUD_Index_set(index_pru(1:Ka));
%         F_set_temp_2 = zeros(GM,Ka*N_BS);
%         for kk_4 = 1:Ka
%             Sel_Index_kk2 = AUD_Index_set(kk_4);
%             F_set_temp_2(:,(kk_4-1)*N_BS+1:kk_4*N_BS) = F_set(:,(Sel_Index_kk2-1)*N_BS+1:Sel_Index_kk2*N_BS);
%         end
%         H_ls = F_set_temp_2\Y_set;
        
        % Update residual
        Res_set = Y_set - F_set_temp_2*H_ls;
        
        % Stopping Criterions
        if iter >= 2 && norm(Res_set(1:end)) > norm(Res_set_pre(1:end))
            AUD_Index_set = AUD_Index_set_iter;
            H_ls = H_ls_iter;
            break;
        else
            AUD_Index_set_iter = AUD_Index_set;
            H_ls_iter = H_ls;
        end
    end
    AUD_Index_K_set = sort(AUD_Index_set);
    
    % Assign estimated complex channel gains
    H_set_hat = zeros(N_BS,P_subc,K);
    for kk_5 = 1:Ka
        H_set_hat(:,:,AUD_Index_set(kk_5)) = H_ls((kk_5-1)*N_BS+1:kk_5*N_BS,:);
    end

end
