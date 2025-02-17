a=2;b=10; %open interval (2,10)
X=a+(b-a)*rand(1,1000000);%simulate uniform RV

[p,edges]=histcounts(X,'Normalization','pdf');%estimated PDF
outcomes = 0.5*(edges(1:end-1) + edges(2:end));%possible outcomes

g=1/(b-a)*ones(1,length(outcomes)); %Theoretical PDF
bar(outcomes,p);hold on;plot(outcomes,g,'r-');
title('Probability Density Function');legend('simulated','theory');
xlabel('Possible outcomes');ylabel('Probability of outcomes');