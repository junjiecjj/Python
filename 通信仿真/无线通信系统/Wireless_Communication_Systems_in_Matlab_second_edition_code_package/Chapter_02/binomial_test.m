n=30; p=1/6; %number of trails and success probability
X = binomialRV(n,p,10000);%generate 10000 bino rand numbers
X_pdf = pdf('Binomial',0:n,n,p); %theoretical probility density
histogram(X,'Normalization','pdf'); %plot histogram
hold on; plot(0:n,X_pdf,'r'); %plot computed theoreical PDF