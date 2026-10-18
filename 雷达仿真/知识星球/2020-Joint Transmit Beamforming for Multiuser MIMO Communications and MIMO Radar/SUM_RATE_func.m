function [SUM_RATE_SDR] = SUM_RATE_func(R_SDR,H,r,M,K,noise_variance)
R_forWr = R_SDR;
for k=1:K
    R_forWr = R_forWr - r(:,:,k);
end
Wc_SDR = zeros(M,K);
for k=1:K
    Wc_SDR(:,k) = (H(k,:)*r(:,:,k)*H(k,:)')^(-1/2)*r(:,:,k)*H(k,:)';
end
Fc_SDR = H*Wc_SDR;
sum_rate_SDR = zeros(1,K);
Z = H*R_forWr*H';
for k=1:K
    sum_rate_SDR(1,k) = Fc_SDR(k,k)^2/(sum(Z(k,:))^2+(sum(Fc_SDR(k,:))-Fc_SDR(k,k))^2+noise_variance);
end
SUM_RATE_SDR = sum(sum_rate_SDR);
end

