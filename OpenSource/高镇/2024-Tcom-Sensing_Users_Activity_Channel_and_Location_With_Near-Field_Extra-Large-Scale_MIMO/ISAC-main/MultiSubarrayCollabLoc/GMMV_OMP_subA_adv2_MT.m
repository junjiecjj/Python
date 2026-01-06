function [AUD_Index_K_set, H_set_hat, Z_Col_Index_set] = GMMV_OMP_subA_adv2_MT(Y_set, Z_set, K, epsilon, M_suba)

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
% Sub_Ind = [];
%% Find AUD support and estimate channels
NorPre=1000;
flag=0;
falggg=0;
while (1)
    % Distributed Correlation
    Corr_set = sum(abs(reshape(pagemtimes(reshape(Res_set,1,GM,P_subc), conj(Z_set)), KNbs,P_subc)) ,2);

    % Find the maximum projection along the different spaces
    Corr_vec = sum(reshape(Corr_set,N_sub,K*M_suba));
    [~, index_ka] = max(Corr_vec);

    % Update the current guess of the common support
    AUD_Index_set = union(AUD_Index_set, index_ka);
    % Project the input signal onto the subspace given by the support using LS
    Z_Col_Index_iter = (index_ka-1)*N_sub+1:index_ka*N_sub;
    Z_Col_Index_set = [Z_Col_Index_set, Z_Col_Index_iter];

    Z_p = Z_set(:,Z_Col_Index_set,:);
    TMP = conj(permute(Z_p,[2,1,3]));
    h00 = pagemtimes(pagemtimes(pageinv(pagemtimes(TMP, Z_p)),TMP), reshape(Y_set,GM,1,P_subc));
    h_p_hat_set = reshape(h00, size(h00,1),P_subc);
    Res_set = Y_set - reshape(pagemtimes(Z_p, h00), GM, P_subc);

    NorNew = norm(Res_set,'fro');
    % Compute the current MSE
    MSE = trace(Res_set'*Res_set)/(GM*P_subc);

    % Compte the number of iteration
    iter_num = iter_num + 1;

    TQQP = ceil(AUD_Index_set/M_suba);
    Index_set = unique(TQQP);
    if MSE <= epsilon
        IDK = hist(TQQP, Index_set);

        if ~ismember(1, IDK)
            falggg = 1;
            break;
        else
            Tmm = find(IDK==1);
            TmmV = Index_set(Tmm);
            if length(TmmV)==1
                % Distributed Correlation
                Corr_set = sum(abs(reshape(pagemtimes(reshape(Res_set,1,GM,P_subc), conj(Z_set)), KNbs,P_subc)) ,2);
                % Find the maximum projection along the different spaces
                Corr_vec = sum(reshape(Corr_set,N_sub,K*M_suba));
                Corrnew = Corr_vec(((TmmV-1)*M_suba+1):TmmV*M_suba);
                [~,indNew]=max(Corrnew);
                index_ka = (TmmV-1)*M_suba+indNew;
                Z_Col_Index_iter = (index_ka-1)*N_sub+1:index_ka*N_sub;
                Z_Col_Index_set = [Z_Col_Index_set, Z_Col_Index_iter];

                Z_p = Z_set(:,Z_Col_Index_set,:);
                TMP = conj(permute(Z_p,[2,1,3]));
                h00 = pagemtimes(pagemtimes(pageinv(pagemtimes(TMP, Z_p)),TMP), reshape(Y_set,GM,1,P_subc));
                h_p_hat_set = reshape(h00, size(h00,1),P_subc);
                Res_set_New = Y_set - reshape(pagemtimes(Z_p, h00), GM, P_subc);

                NorNew_New = norm(Res_set_New,'fro');
                if NorNew_New < NorNew
                    flag = 1;
                    break;
                end
            end
            if flag==0
                kkk = [];
                for i=1:length(Tmm)
                    kkk = [kkk, find(TQQP == TmmV(i))];
                end
                AUD_Index_set = setdiff(AUD_Index_set, AUD_Index_set(kkk));
                break;
            end
        end
    end

    if NorNew >=NorPre
        break;
    end
    NorPre = NorNew;

end

AUD_Index_K_set = unique(ceil(AUD_Index_set/M_suba));  % active UE set

if flag==0 && falggg==0
    Z_Col_Index_set = [];
    for i=1:length(AUD_Index_set)
        index_ka = AUD_Index_set(i);
        Z_Col_Index_iter = (index_ka-1)*N_sub+1:index_ka*N_sub;
        Z_Col_Index_set = [Z_Col_Index_set, Z_Col_Index_iter];
    end
    % Z_active_set = Z_set(:,Z_Col_Index_set);
    % H_AUD_hat_set = pinv(Z_active_set)*Y_set;

    Z_p = Z_set(:,Z_Col_Index_set,:);
    TMP = conj(permute(Z_p,[2,1,3]));
    h00 = pagemtimes(pagemtimes(pageinv(pagemtimes(TMP, Z_p)),TMP), reshape(Y_set,GM,1,P_subc));
    h_p_hat_set = reshape(h00, size(h00,1),P_subc);

    % h_p_hat_set = zeros(size(Z_Col_Index_set,2),P_subc);
    % for pp_2 = 1:P_subc
    %     Z_Col_Index_p = Z_set(:,Z_Col_Index_set,pp_2);
    %     h_p_hat_set(:,pp_2) = pinv(Z_Col_Index_p)*Y_set(:,pp_2);
    % end
end
%% Assign estimated complex channel gains
H_set_hat = zeros(N_BS*K,P_subc);
H_set_hat(Z_Col_Index_set,:)=h_p_hat_set;
H_set_hat = reshape(H_set_hat, [N_BS,K,P_subc]);
H_set_hat = permute(H_set_hat,[1,3,2]);
end