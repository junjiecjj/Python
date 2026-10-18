function [INR] = INR_func(R_SDR,H,r,M,K,noise_variance)
R_forWr = R_SDR;
for k=1:K
    R_forWr = R_forWr - r(:,:,k);
end
Wc_SDR = zeros(M,K);
for k=1:K
    Wc_SDR(:,k) = (H(k,:)*r(:,:,k)*H(k,:)')^(-1/2)*r(:,:,k)*H(k,:)';
end
Fc_SDR = H*Wc_SDR;
INR = (norm(Fc_SDR,2)^2-norm(diag(Fc_SDR),2)^2)/noise_variance;
end

