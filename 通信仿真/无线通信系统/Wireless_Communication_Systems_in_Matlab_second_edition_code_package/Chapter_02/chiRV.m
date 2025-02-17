function Y= chiRV(k,N)
%Central Chi distributed univariate random numbers
% k=degrees of freedom, N=Number of samples
Y=0;
for j=1:k,
%Sum of square of k independent standard normal Random Variables
    Y= Y + randn(1,N).^2;
end
Y=sqrt(Y); %Taking Square-root
end