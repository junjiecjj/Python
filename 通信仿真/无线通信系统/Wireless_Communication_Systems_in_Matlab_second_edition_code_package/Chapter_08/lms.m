function w=lms(N,mu,r,a)
%Function implementing LMS update equations
% N = desired length of the filter
% mu = step size for the LMS update
% r = received/input signal
% a = reference signal
%
%Returns: % w = optimized filter coefficients

w=zeros(1,N);%initialize filter taps to zeros
for k=N:length(r)   
    r_vector = r(k:-1:k-N+1);
    e = a(k)-w *r_vector';
    w = w + mu*e*r_vector;
end
end