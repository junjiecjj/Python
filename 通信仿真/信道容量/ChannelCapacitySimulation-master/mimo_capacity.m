function cap = mimo_capacity(N,M,SNR)

SNR_D = 10^(SNR*0.1); %SNR in decimal
C = zeros(1,3000);

for i = 1:1:3000
    H = raylrnd(1/sqrt(2),M,N) + 1i * raylrnd(1/sqrt(2),M,N);
    [U,S,V] = svd(H);
    d = diag(S);
    C_temp = zeros(1,length(d));
    for j = 1:1:length(d)
        C_temp(1,j) = log2(1+d(j,1)^2*SNR_D/N);
    end
    C(1,i) = sum(C_temp);
end

cap = mean(C);