function [SNRdB, MSE_OAMP, BER_OAMP] = LDPC_SE_EXIT(snr, Nu, dia,variable_edge_degree1, variable_edge_weight1, ...
    parity_edge_degree1, parity_edge_weight1,variable_edge_degree2, variable_edge_weight2,...
    parity_edge_degree2, parity_edge_weight2)

SNRdB = 10*log10(snr);
SNRdB = (SNRdB-1):0.1:(SNRdB+3);
Iteration = 1000;
BER_OAMP = zeros(Iteration,length(SNRdB));
MSE_OAMP = zeros(Iteration,length(SNRdB));

for iii = 1:length(SNRdB)    
    Sigma_n = 10^(-SNRdB(iii)/10);
    v_LDPC1= 1;
    v_LDPC2= 1;
    I_memory_OAMP1 = 0;
    I_memory_OAMP2 = 0;
    ite_inner = 1;
    for it = 1:Iteration
        %% NLD + LD
        v_OAMP1= OAMP(v_LDPC1, Nu, dia, Sigma_n);
        v_OAMP2= OAMP(v_LDPC2, Nu, dia, Sigma_n);
        %% LDPC EXIT
        [~, v_post_out1, I_memory_OAMP1, BER_out1]= LDPC_SE_memory(v_OAMP1, I_memory_OAMP1, ite_inner, variable_edge_degree1, variable_edge_weight1, parity_edge_degree1, parity_edge_weight1, 1e-6);
        [~, v_post_out2, I_memory_OAMP2, BER_out2]= LDPC_SE_memory(v_OAMP2, I_memory_OAMP2, ite_inner, variable_edge_degree2, variable_edge_weight2, parity_edge_degree2, parity_edge_weight2, 1e-6);
        
%         [v_LDPC, I_memory_OAMP1, I_memory_OAMP2, BER_OAMP(it,iii)] = LDPC_SE_memory_asym(v_OAMP, I_memory_OAMP1, I_memory_OAMP2, ite_inner, variable_edge_degree1, variable_edge_weight1, ...
%             parity_edge_degree1, parity_edge_weight1, variable_edge_degree2, variable_edge_weight2, parity_edge_degree2, parity_edge_weight2, 1e-6);        
        MSE_OAMP(it,iii) = (v_post_out1+v_post_out1)/2;
        %snr_LDPC = ((v_post_out1+v_post_out1)/2)^-1;
        %snr_LDPC1 = (v_post_out1^-1-v_OAMP1^-1)^-1;
        %snr_LDPC2 = (v_post_out2^-1-v_OAMP2^-1)-1;
        v_LDPC1 = v_post_out1
        v_LDPC2 = v_post_out2
        I_memory_OAMP1=I_memory_OAMP1
        I_memory_OAMP2=I_memory_OAMP2
    end
end

%%
 
function v_out =OAMP(x, Nu, dia, Sigma_n)
    
phi_1 = Phi_Transfer(x, Nu, dia.^2, Sigma_n);
v_out = phi_1^-1;
%phi_1 = Phi_Transfer(1, Nu, dia.^2, Sigma_n);
% if x<=phi_1
%     v_out = MMSE_QPSK(x);
% else
%     v_lmmse = LMMSE(x, Nu, dia, Sigma_n);
%     v_QPSK = MMSE_QPSK(x);
%     v_out =min(v_lmmse,v_QPSK);
% end
end

function v_out = LMMSE(rho, Nu, dia, Sigma_n)
    
precious = 10^-5;    
v1 = 10^-20*ones(size(rho));
v2 = ones(size(rho));
dia2=dia.^2;
while (norm(v1-v2)>precious)
    v = (v1+v2)/2;
    t = Phi_Transfer(v, Nu, dia2, Sigma_n)- rho;
    v1(t>0) = v(t>0);  
    v2(t<=0) = v(t<=0);  
end
v_out = 1 ./ (rho + 1./v2);

end

%%
function rho = Phi_Transfer(v, Nu, dia2, Sigma_n)
% v and rho are 1xK
% dia is row 1xNu
rho =  v.^-1 .* ( Nu./sum(1./( Sigma_n^-1 * dia2' * v+ 1) ) -1) ;
end

%%
function v = MMSE_QPSK(snr)
%snr = snr/2;   % scale from BPSK for QPSK
max=100;
v = zeros(size(snr));
for i=1:length(snr)
    v(i) = 1 - integral(@(x) f_QPSK(x,snr(i)), -max, max);
end

%% 
function y = f_QPSK(x,snr) 
y  = exp(-x.^2/2)/sqrt(2*pi) .* tanh(snr - sqrt(snr).*x);
end
end
end