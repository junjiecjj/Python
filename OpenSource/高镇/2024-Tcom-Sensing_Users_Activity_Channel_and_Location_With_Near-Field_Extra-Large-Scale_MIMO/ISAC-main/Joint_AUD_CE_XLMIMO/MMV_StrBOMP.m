function [AUD_Index_K_set, H_set_hat, Z_Col_Index_set] = MMV_OMP_subA_adv(Y_set, Z_set, K, epsilon, M_suba)

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
while (1)
    % Distributed Correlation
    Corr_set = sum(abs(Z_set'*Res_set),2);
    
    % Find the maximum projection along the different spaces
    Corr_vec = sum(reshape(Corr_set,N_sub,K*M_suba));
    [~, index_ka] = max(Corr_vec);
    
    % Update the current guess of the common support
    AUD_Index_set = union(AUD_Index_set, index_ka);
    
    % Project the input signal onto the subspace given by the support using LS
    Z_Col_Index_iter = (index_ka-1)*N_sub+1:index_ka*N_sub;
    Z_Col_Index_set = [Z_Col_Index_set, Z_Col_Index_iter];


    Z_active_set = Z_set(:,sort(Z_Col_Index_set));
    H_AUD_hat_set = pinv(Z_active_set)*Y_set;
    PowCh = sum(reshape(sum(abs(H_AUD_hat_set).^2,2)/P_subc,N_sub,length(AUD_Index_set)),1)/N_sub;
    % Update residual
    Res_set = Y_set - Z_active_set*H_AUD_hat_set;
    
    % Compute the current MSE
    MSE = trace(Res_set'*Res_set)/(GM*P_subc);
    
    % Compte the number of iteration
    iter_num = iter_num + 1;

    TQQP = ceil(AUD_Index_set/M_suba);
    Index_set = unique(TQQP);
    if MSE <= epsilon
        IDK = hist(TQQP, Index_set);
        
        if ~ismember(1, IDK)
            break;
        else
            % PwUE = zeros(1,length(IDK));
            % jj=1;
            % for ii = 1:length(IDK)
            %     PwUE(ii) = sum(PowCh(jj:jj+IDK(ii)-1));
            %     jj = jj+IDK(ii);
            % end
            % plot(PwUE);
            % hold on;
            % plot(1:length(IDK), ones(1,length(IDK))*mean(PwUE));
            % close;

            Tmm = find(IDK==1);
            TmmV = Index_set(Tmm);
            kkk = [];
            for i=1:length(Tmm)
                kkk = [kkk, find(TQQP == TmmV(i))];      
            end
            % PowCh
            % PowCh(kkk)
            AUD_Index_set = setdiff(AUD_Index_set, AUD_Index_set(kkk));
            break;
        end
    end
    NorNew = norm(Res_set,'fro');
    if NorNew >=NorPre
        break;
    end
    NorPre = NorNew;

end

AUD_Index_K_set = unique(ceil(AUD_Index_set/M_suba));  % active UE set

Z_Col_Index_set = [];
for i=1:length(AUD_Index_set)
    index_ka = AUD_Index_set(i);
    Z_Col_Index_iter = (index_ka-1)*N_sub+1:index_ka*N_sub;
    Z_Col_Index_set = [Z_Col_Index_set, Z_Col_Index_iter];
end
Z_active_set = Z_set(:,Z_Col_Index_set);
H_AUD_hat_set = pinv(Z_active_set)*Y_set;

%% Assign estimated complex channel gains
H_set_hat0 = zeros(N_BS*K,P_subc);
H_set_hat0(Z_Col_Index_set, :) = H_AUD_hat_set;
H_set_hat = reshape(H_set_hat0, N_BS, K, P_subc);
H_set_hat = permute(H_set_hat, [1, 3, 2]);
end