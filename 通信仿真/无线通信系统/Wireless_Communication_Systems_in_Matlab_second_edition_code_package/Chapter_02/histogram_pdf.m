%using random function
mu=0;sigma=1;%mean=0,deviation=1
L=100000; %length of the random vector
R = random('Normal',mu,sigma,L,1);%method 1

%using randn function
mu=0;sigma=1;%mean=0,deviation=1
L=100000; %length of the random vector
R = randn(L,1)*sigma + mu; %method 2

%Box-Muller transformation
mu=0;sigma=1;%mean=0,deviation=1
L=100000; %length of the random vector
U1 = rand(L,1); %uniformly distributed random numbers U(0,1)
U2 = rand(L,1); %uniformly distributed random numbers U(0,1)
Z = sqrt(-2*log(U1)).*cos(2*pi*U2);%Standard Normal distribution
R = Z*sigma+mu;%Normal distribution with mean and sigma

histogram(R,'Normalization','pdf'); %plot estimated pdf from data

X = -4:0.1:4; %range of x to compute the theoretical pdf
fx_theory = pdf('Normal',X,mu,sigma); %theoretical pdf
hold on; plot(X,fx_theory,'r'); %plot computed theoretical PDF
title('Probability Density Function'); xlabel('values - x'); 
ylabel('pdf - f(x)'); axis tight; legend('simulated','theory');

numBins=50; %choose appropriately
%use hist function and get unnormalized values
[f,x]=hist(R,numBins); 
figure; plot(x,f/trapz(x,f),'b-*');%normalized histogram from data

X = -4:0.1:4; %range of x to compute the theoretical pdf
fx_theory =   pdf('Normal',X,mu,sigma); %theoretical pdf
hold on; plot(X,fx_theory,'r'); %plot computed theoretical PDF
title('Probability Density Function'); xlabel('values - x'); 
ylabel('pdf - f(x)'); axis tight; legend('simulated','theory');