[f,x]=hist(1+2.*randn(10000,1),100); %Simulated PDF
bar(x,f/trapz(x,f));hold on;
g=1/sqrt(2*pi*4)*exp(-0.5*(x-1).^2/(4)); %Theoretical PDF
plot(x,g,'r');hold off;
title('Probability Density Function');legend('simulated','theory');
xlabel('Possible outcomes');ylabel('Probability of outcomes');