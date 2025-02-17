function y = conv_brute_force(x,h)
%Brute force method to compute convolution

N=length(x); M=length(h);
y = zeros(1,N+M-1);
for i = 1:N,
   for j = 1:M,
       y(i+j-1) = y(i+j-1) + x(i) * h(j);
   end
end
end