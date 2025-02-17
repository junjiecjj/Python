function Y= nakagamiRV(m,w,N)
%Generate univariate Nakagami distributed random numbers
% m=parameter to control shape, w=parameter to control spread
% N=number of samples to generate
Y=sqrt(w/(2*m))*chiRV(2*m,N);
end