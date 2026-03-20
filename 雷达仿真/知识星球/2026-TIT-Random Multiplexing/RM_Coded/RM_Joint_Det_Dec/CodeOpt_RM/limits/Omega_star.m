function [rho_lmmse_dB] = Omega_star(K, M, snr, kappa)

T = min(M,K);
dia = kappa.^(-[0:T-1]' / T);
dia = sqrt(K) * dia / norm(dia);  % add 0 into dia if M<K


%% linear mmse
v = 10.^[-10:0.1:0];
rho_lmmse = LMMSE_Div(v, dia, K, snr^-1); % LMMSE transfer curve
rho_lmmse_dB = 10*log10(rho_lmmse);
semilogy(rho_lmmse_dB,v,'blue-'); 
xlabel('\rho (dB)');
ylabel('v'); 
hold on;
%% non-linear mmse
rho_dB =-10:0.1:20;
rho = 10.^(rho_dB/10);
v_mmse = MMSE_QPSK_Div(rho); % QPSK demodulation curve
semilogy(rho_dB,v_mmse,'red-'); %semilogy plot

end

