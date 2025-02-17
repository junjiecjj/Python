%Test and Demonstrate Central Chi Distribution
clear;clc;
k=[1,2,3,4,5]; %degrees of freedom
N = 1000000; %Number of Samples
bins = 0:0.01:10; %bins for plotting histogram

%allocating space for Chi RVs and distributions
Y = zeros(length(k),N);
binPos= zeros(length(k),length(bins));
dist= zeros(length(k),length(bins));
lineColors=['r','g','b','c','m','y','k']; %line color arguments
legendString ={}; %cell array of string for printing legend

%Generating Chi Distribution with k degrees of freedom
for i=1:length(k), %outer loop - runs for each degrees of freedom

Y(i,:)= chiRV(k(i),N);

%Computing Histogram for each degrees of freedom
[Q,X] = hist(Y(i,:),bins);
dist(i,:)  =Q;
binPos(i,:)=X;

%plot commands 
plot(binPos(i,:),dist(i,:)/(0.01*N),lineColors(i),'LineWidth',1.5);
hold on;
legendString{i}=strcat(' k=',num2str(i));
end
legend(legendString);
title('Probability Density Function - central Chi Distribution ','fontsize',12);
xlabel('Random Variable - x','fontsize',12);ylabel('f_X(x)','fontsize',12);axis([0 4 0 1]);