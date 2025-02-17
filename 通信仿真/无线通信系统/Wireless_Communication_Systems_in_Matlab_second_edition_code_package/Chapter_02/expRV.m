function T = expRV(lambda,L)
%Generate random number sequence that is exponentially distributed
%lambda - rate parameter, L - length of the sequence generated
U = rand(1,L); %continuous uniform random numbers in (0,1)
T = -1/lambda*(log(1-U));
end