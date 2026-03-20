function S = PSK_Constellation(m)
% zero mean and unit variance
m = floor(m); 
S =zeros(1,m);

for k=1:m
   S(k) = exp(1i*2*pi*k/m); 
end

end