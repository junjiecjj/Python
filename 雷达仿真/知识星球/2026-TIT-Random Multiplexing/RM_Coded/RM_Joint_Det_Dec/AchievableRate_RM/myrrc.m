function b = myrrc(beta,t1,fs,sps)
t=t1*fs;
if abs(1-(2*beta*t).^2)>sqrt(eps)
    b = sinc(t).*(cos(pi*beta*t))./(1-(2*beta*t).^2)/sps;
else
    b = beta*sin(pi/(2*beta))/(2*sps);
end