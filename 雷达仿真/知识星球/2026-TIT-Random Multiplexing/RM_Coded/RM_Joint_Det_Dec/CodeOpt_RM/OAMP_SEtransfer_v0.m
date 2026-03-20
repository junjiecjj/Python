function [vstar1, rho1, vstar2, rho2, v_final,rho_final,v_star,rho_star] = OAMP_SEtransfer(M, K, kappa,snr)

%generate SE transfer function.
T = min(M,K);
dia = kappa.^(-[0:T-1]' / T);
dia = sqrt(K) * dia / norm(dia);  % add 0 into dia if M<K

%linear mmse
ld_v = 10.^[-10:0.05:0];
ld_rho= LMMSE_Div(ld_v, dia, K, snr.^-1); % LMMSE transfer curve
%non-linear mmse
nld_rho_dB =-20:0.1:10;
nld_rho = 10.^(nld_rho_dB/10);
nld_v = MMSE_QPSK_Div(nld_rho); % QPSK demodulation curve

%%
rho_star = 0.9407;
v_star = 0.4688;
rho_max  = snr;
v_min = 1e-10
%%
step1= (rho_star/20);
rho1 = step1:step1:rho_star;
vstar1 = MMSE_QPSK_Div(rho1);

step2= ((v_star-v_min)/20);
v_2 = (v_star-step2):-step2:(v_min+step2);

step3= step2/100;
v_3 = (v_min+step2-step3):-step3:(v_min+step3);

step4= step3/10;
v_4 = (v_min+step3-step4):-step4:v_min;

vstar2  = [v_2, v_3, v_4];
rho2 = LMMSE_Div(vstar2, dia, K, snr.^-1); 

v_final = [vstar1, vstar2];
rho_final=[rho1,rho2];

end

