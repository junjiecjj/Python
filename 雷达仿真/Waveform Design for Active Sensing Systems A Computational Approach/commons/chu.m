function x = chu(N)
% x = chu(N), generate the Chu sequence of length N

if mod(N,2)==0
    x = exp(1i * pi * (1:N).^2 /N);
else
    x = exp(1i * pi * (2:(N+1)) .* (1:N) /N);
end

x = x.';