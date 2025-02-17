alpha = 1; %valid values -2<=alpha<=2.
N = 63; %The filter order of the implemented IIR filter
nSamp = 10000; %Number of noise samples to generate

a=[1 zeros(1,N)];%AR coefficients [a_0,...,a_N]
for k=2:N+1,
   a(k)= ( k - 2 - alpha/2 )*a(k-1)/(k-1);
end

w = randn(1,nSamp);%zero-mean white noise samples
y = filter(1,a,w); %filter white noise though the AR filter
%The Welch PSD estimate and theoretical PSD of the colored noise
[Pyy,F]=plotWelchPSD(y,1);%estimated PSD
psdTheory = 1./(F(2:end).^alpha);%Theoretical PSD

figure; plot(y); title('Colored noise'); %Plot the results
xlabel('sample index [n]');ylabel('y[n]');
figure;plot(log2(F(2:end)),10*log10(Pyy(2:end)),'k');hold on;
plot(log2(F(2:end)),10*log10(psdTheory),'r','linewidth',2)
xlabel('log_2(F)(Hz)'); ylabel('P_{yy} (dB)');  grid on;
title('PSD of Colored noise');legend('PSD estimate','Theoretical PSD');