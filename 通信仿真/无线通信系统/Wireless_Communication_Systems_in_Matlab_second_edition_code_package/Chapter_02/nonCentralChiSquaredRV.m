function X = nonCentralChiSquaredRV(k,lambda,N)
%Generate non-central Chi-squared distributed random numbers
%k - degrees of freedom, lambda - non-centrality parameter
%N - number of samples
X=(sqrt(lambda)+randn(1,N)).^2;%one normal RV nonzero mean & var=1
for i=1:k-1, %k-1 normal RVs with zero mean & var=1
    X = X + randn(1,N).^2;
end
