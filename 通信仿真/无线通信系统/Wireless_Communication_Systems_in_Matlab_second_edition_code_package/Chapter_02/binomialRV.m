function X = binomialRV(n,p,L)
%Generate Binomial random number sequence
%n - the number of independent Bernoulli trials
%p - probability of success yielded by each trial
%L - length of sequence to generate
X = zeros(1,L);
for i=1:L,
    X(i) = sum(bernoulliRV(n,p));
end
end