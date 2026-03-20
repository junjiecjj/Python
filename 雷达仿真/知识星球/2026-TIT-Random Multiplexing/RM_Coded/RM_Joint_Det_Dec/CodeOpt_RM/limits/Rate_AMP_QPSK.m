%%
function R_AMP = Rate_AMP_QPSK(beta, snr)

%max=100;
R_AMP = zeros(length(beta), length(snr));
%R_min = zeros(length(beta), length(snr));
for i=1:length(beta)
     %fprintf('beta = %d \n',beta(i));
    for j=1:length(snr)
%         rho = Rho_QPSK(beta(i), snr(j),precise);
%         %tem = rho - integral2(@(x,snr) f_QPSK(x,snr),-max, max,0,rho);
%          tem = integral(@(x) MMSE_QPSK(x),0,rho);
%         R_AMP(i,j) = beta(i)^-1*(rho./snr(j) - log(rho./snr(j)) -1) + tem;
        
        R_AMP(i,j) = quad(@(x) Min_stransfer(x,snr(j),beta(i)),0,snr(j));
    end
end
  
R_AMP =R_AMP/log(2);
%%
function v = Min_stransfer(x,snr,beta)
v = min(beta^-1*(x.^-1-snr^-1), MMSE_QPSK(x));
end
end