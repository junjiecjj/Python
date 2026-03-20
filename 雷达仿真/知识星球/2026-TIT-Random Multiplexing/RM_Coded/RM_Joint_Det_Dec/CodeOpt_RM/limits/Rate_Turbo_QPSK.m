%%
function R_Turbo = Rate_Turbo_QPSK(beta, snr)

MAX=100;
R_Turbo = zeros(length(beta), length(snr));

for i=1:length(beta)
     %fprintf('beta = %d \n',beta(i));
    for j=1:length(snr)
        R_Turbo(i,j) = log(4) - quad(@(x) f_fun(x,snr(j),beta(i)), 0, MAX);
    end
end
  
R_Turbo = R_Turbo/log(2);

%%
%snr=1000;
%beta=1;
%plot([0:0.01:100],f_fun([0:0.01:100], snr, beta));
function y = f_fun(x,snr,beta)
y = MMSE_QPSK( x + f_phi(MMSE_QPSK(x), snr, beta) );
end
%%
function y = f_phi(x,snr,beta)
    
ssigma = snr^-1;
tem = ssigma + (beta -1)*x;
y = 2 ./ (tem+sqrt(tem.^2 + 4*ssigma*x));

end
end

%%
function v = MMSE_QPSK(snr)

max=100;
v = zeros(size(snr));
for i=1:length(snr)
    v(i) = 1 - quad(@(x) f_QPSK(x,snr(i)), -max, max);
end
end

function y = f_QPSK(x,snr) 
y  = exp(-x.^2/2)/sqrt(2*pi) .* tanh(snr - sqrt(snr).*x);
end