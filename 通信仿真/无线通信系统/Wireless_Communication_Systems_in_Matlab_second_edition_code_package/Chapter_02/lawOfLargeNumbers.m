n=1000; %number of trials
p=0.7; %probability of success
X=bernoulliRV(n,p); %Bernoulli random variable
y_sum=sum(triu(repmat(X,[prod(size(X)) 1])')); %cumulative sum
avg = y_sum./(1:1:n); %average of results
plot(1:1:n,avg,'.'); hold on;
xlabel('Trial #'); ylabel('Probability of Heads');
plot(p*ones(1,n),'r'); legend('average','expected');