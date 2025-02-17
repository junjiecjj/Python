X=randi(6,100000,1); %Simulate throws of dice,S={1,2,3,4,5,6}
[pmf,edges]=histcounts(X,'Normalization','pdf');%estimated PMF
outcomes = 0.5*(edges(1:end-1) + edges(2:end));%S={1,2,3,4,5,6}
g=1/6*ones(1,6); %Theoretical PMF

bar(outcomes,pmf);hold on;stem(outcomes,g,'r-');
title('Probability Mass Function');legend('simulated','theory');
xlabel('Possible outcomes');ylabel('Probability of outcomes');