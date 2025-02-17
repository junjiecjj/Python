%Test and Demonstrate Central Chi Distribution
clearvars; clc;
k=[1,2,3,4,5];N = 10e5;  %degrees of freedom & Number of Samples

lineColors=['r','g','b','c','m','y','k']; %line color arguments
legendString =cell(1,5); %cell array of string for printing legend
figure;hold on;
%Generating Chi Distribution with k degrees of freedom
for i=1:length(k), %outer loop - runs for each degrees of freedom
    Y= chiRV(k(i),N);    
    [Q,X] = hist(Y,1000);%Histogram for each degree of freedom
    plot(X,Q/trapz(X,Q),lineColors(i));
    legendString{i}=strcat(' k=',num2str(i));
end
legend(legendString);
title('Central Chi Distributed RV - PDF','fontsize',12);
xlabel('Parameter - x');ylabel('f_X(x)');axis([0 4 0 1]);