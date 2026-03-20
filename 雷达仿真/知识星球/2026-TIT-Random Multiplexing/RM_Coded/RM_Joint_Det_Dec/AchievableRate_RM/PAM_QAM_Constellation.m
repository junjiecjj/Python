function [S_PAM, S_QAM] = PAM_QAM_Constellation(m)
% zero mean and unit variance
n = floor(m^2);
S_PAM =zeros(1,m);
S_QAM = zeros(1,n);

for k=1:m
   S_PAM(k) = (2*k-1-m) * sqrt(3/(m^2-1)); 
end

for k = 1:m
    for z = 1:m
        index = (z-1) * m + k;
        S_QAM(index) = (S_PAM(k)+ S_PAM(z)*1i)/sqrt(2); 
    end
end

end