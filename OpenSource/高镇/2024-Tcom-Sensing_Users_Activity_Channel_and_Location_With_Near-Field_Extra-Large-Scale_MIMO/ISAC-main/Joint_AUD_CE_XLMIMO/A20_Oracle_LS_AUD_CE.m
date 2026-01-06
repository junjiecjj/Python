function H_set_hat = A20_Oracle_LS_AUD_CE(Y_set, Z_set, K, Ka, AUD_Index_K_set)

[GM,P_subc] = size(Y_set);
N_BS = size(Z_set,2)/K;

%% Estimate channels
Z_active_set = zeros(GM,Ka*N_BS);
for kk = 1:Ka
    index_ka = AUD_Index_K_set(kk);
    Z_active_set(:,(kk-1)*N_BS+1:kk*N_BS) = Z_set(:,(index_ka-1)*N_BS+1:index_ka*N_BS);
end
H_AUD_hat_set = pinv(Z_active_set)*Y_set;

%% Assign estimated complex channel gains
H_set_hat = zeros(N_BS,P_subc,K);
for kk_2 = 1:Ka
    H_set_hat(:,:,AUD_Index_K_set(kk_2)) = H_AUD_hat_set((kk_2-1)*N_BS+1:kk_2*N_BS,:); %
end

end
