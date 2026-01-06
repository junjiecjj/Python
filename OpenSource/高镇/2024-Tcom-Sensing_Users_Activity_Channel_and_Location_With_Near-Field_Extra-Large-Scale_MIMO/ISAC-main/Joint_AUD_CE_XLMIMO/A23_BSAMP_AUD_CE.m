function [AUD_Index_K_set, H_set_hat] = A23_BSAMP_AUD_CE(Y_set, Z_set, K, Ka)


    [GM, P_subc] = size(Y_set);
    K_Nbs = size(Z_set,2);
    N_BS = K_Nbs/K;
    
    %% Initializations
    AUD_Index_set = [];	% support set
    Res_set = Y_set;	% residual signals
    S = 1;
    L = S; % Size of the finalist in the first stage
    Stage = 1; 
    IterMax = GM;
    for ii = 1:IterMax
        % (1) Preliminary Test
        Corr_set = Z_set'*Res_set;
        Corr_vec = zeros(K,1);
        for kk_1 = 1:K
            Corr_vec(kk_1) = sum(sum(abs(Corr_set((kk_1-1)*N_BS+1:kk_1*N_BS,:))));
        end
        [~, index_sel_1] = sort(Corr_vec, 'descend');
        index_sel_L = index_sel_1(1:L);
        
        % (2) Make Candidate List
        Cand_index_set = union(AUD_Index_set, index_sel_L);
        
        % (3) Final Test
        Cand_Ka_est = length(Cand_index_set);
        Z_set_temp1 = zeros(GM,Cand_Ka_est*N_BS);
        for kk_2 = 1:Cand_Ka_est
            Z_set_temp1(:,(kk_2-1)*N_BS+1:kk_2*N_BS) = Z_set(:,(Cand_index_set(kk_2)-1)*N_BS+1:Cand_index_set(kk_2)*N_BS);
        end
        H_ls_temp = pinv(Z_set_temp1)*Y_set;

        % max(max(abs(Z_set_temp1\Y_set-pinv(Z_set_temp1)*Y_set)))

        % Support Pruning
        Prun_vec = zeros(Cand_Ka_est,1);
        for kk_3 = 1:Cand_Ka_est
            Prun_vec(kk_3) = sum(sum(abs(H_ls_temp((kk_3-1)*N_BS+1:kk_3*N_BS,:))));
        end
        [~, index_pru] = sort(Prun_vec, 'descend');
        Fin_Index_set = Cand_index_set(index_pru(1:L));
        
        % (4) Estimation and Compute Residue
        Z_set_temp2 = zeros(GM,L*N_BS);
        for kk_4 = 1:L
            Sel_Index_kk2 = Fin_Index_set(kk_4);
            Z_set_temp2(:,(kk_4-1)*N_BS+1:kk_4*N_BS) = Z_set(:,(Sel_Index_kk2-1)*N_BS+1:Sel_Index_kk2*N_BS);
        end
        H_ls = pinv(Z_set_temp2)*Y_set;
        
        % Update residual
        Res_set_new = Y_set - Z_set_temp2*H_ls;
        
        % Stopping Criterions
        if numel(Fin_Index_set) >= Ka % trace(Res_set_new'*Res_set_new)/(GM*P_subc) < epsilon
            AUD_Index_set = Fin_Index_set;
            break; % quit the iteration
        elseif norm(Res_set_new) >= norm(Res_set) % stage switching 
            Stage = Stage + 1; % Update the stage index 
            L = Stage*S; % Update the size of finalist
        else
            AUD_Index_set = Fin_Index_set; % Update the finalist index set
            Res_set = Res_set_new; % Update the residue
        end
    end
    AUD_Index_K_set = sort(AUD_Index_set);
    
    % Assign estimated complex channel gains
    H_set_hat = zeros(N_BS,P_subc,K);
    for kk_5 = 1:numel(AUD_Index_set)
        H_set_hat(:,:,AUD_Index_set(kk_5)) = H_ls((kk_5-1)*N_BS+1:kk_5*N_BS,:);
    end

end
