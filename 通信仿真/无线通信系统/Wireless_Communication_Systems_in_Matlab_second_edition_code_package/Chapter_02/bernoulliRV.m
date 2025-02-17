function X = bernoulliRV(L,p)
%Generate Bernoulli random number with success probability p
%L is the length of the sequence generated
U = rand(1,L); %continuous uniform random numbers in (0,1)
X = (U<p);
end