function y = Turbo_phi(x,snr,beta)
    
ssigma = snr^-1;
tem = ssigma + (beta -1)*x;
y = 2 ./ (tem+sqrt(tem.^2 + 4*ssigma*x));

end