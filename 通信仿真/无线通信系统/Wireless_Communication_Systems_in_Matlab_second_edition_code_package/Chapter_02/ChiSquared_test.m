%Test and Demonstrate Central Chi Squared Distribution
clearvars; clc;
k=[1,2,3,4,5]; N = 10e5;%degrees of freedom & Number of Samples
 
lineColors=['r','g','b','k','m']; %line color arguments
legendString =cell(1,5); %cell array of string for printing legend
%Generating Chi-Square Distribution with k degrees of freedom
for i=1:length(k), %outer loop - runs for each degrees of freedom
    Y= chisquaredRV(k(i),N);
    %Computing Histogram for each degree of freedom
    [Q,X] = hist(Y,1000);
    plot(X,Q/trapz(X,Q),lineColors(i)); %normalized histogram
    hold on; legendString{i}=strcat(' k=',num2str(i));
end
legend(legendString);xlabel('Parameter - x');ylabel('f_X(x)');
title('Central Chi Squared RV - PDF');axis([0 8 0 0.5]);