lambda=1.5; L=100000; T = expRV(lambda,L);%simulating the RV

t_range=0:0.2:5;
T_pdf = lambda*exp(-lambda*t_range); %theoretical PDF
histogram(T,'Normalization','pdf'); hold on;%histogram from data
plot(t_range,T_pdf,'r'); %plot computed theoreical PDF
xlabel('t');ylabel('pdf - f_T(t)'); 
title(['Exponential PDF - \lambda=',num2str(lambda)]);