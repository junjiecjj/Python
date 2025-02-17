function Y= chisquaredRV(k,N)
%Generate Chi-Squared Distributed univariate random numbers
% k=degrees of freedom, N=Number of samples
Y=0;
for j=1:k, %Sum of square of k independent standard normal rvs
    Y= Y + randn(1,N).^2;
end