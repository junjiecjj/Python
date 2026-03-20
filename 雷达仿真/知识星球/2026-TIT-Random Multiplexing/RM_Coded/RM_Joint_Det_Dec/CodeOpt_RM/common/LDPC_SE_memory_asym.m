function [v_post_out, I_memory_out1, I_memory_out2, BER_out]= LDPC_SE_memory_asym(v_in, I_memory1, I_memory2, ite, dv1, b1, dc1, d1, dv2, b2, dc2, d2, threshold)
%% Note:
% calculate v_i for i-th LDPC codes
% average v=avg(sum(v_i)) to obtain input v of LMMSE


%% LDPC Code1
[~, v_post_out1, I_memory_out1, BER_out1]= LDPC_SE_memory(v_in, I_memory1, ite, dv1, b1, dc1, d1, threshold)

%% LDPC Code2
[~, v_post_out2, I_memory_out2, BER_out2]= LDPC_SE_memory(v_in, I_memory2, ite, dv2, b2, dc2, d2, threshold)

%%
v_post_out = (v_post_out1+v_post_out2)/2;
BER_out = (BER_out1 + BER_out2)/2;
end
