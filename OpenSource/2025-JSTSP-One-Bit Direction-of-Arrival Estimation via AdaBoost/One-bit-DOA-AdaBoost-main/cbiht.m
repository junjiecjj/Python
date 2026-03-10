function [DOA_deg,DOA_rad] = cbiht(x0_one,P,K,M)


tt= 1:P;
thet = pi - (2*pi*(tt-1))/P ;
GG = length(thet);
Ahat = zeros(M,GG);
idxR = (0 : (M - 1))';
for k = 1:GG
        Ahat(:,k) = exp(1j*thet(k)*idxR);
end
S_0 = Ahat'*x0_one;
miu = norm(Ahat)^(-2);

%% CBIHT

for ii=1:100
    Z= Ahat*S_0;
    Y = sign(real(Z)) + 1j*sign(imag(Z)) - x0_one;
    W = S_0 - miu*(Ahat')*Y;
    gg = zeros(GG,1);
    for jj=1:GG
        gg(jj) = norm(W(jj,:));
    end
    [~,I] = findpeaks(gg,'SortStr','descend','NPeaks',K);
    S_0 =zeros(size(W));
    for jj=1:K
        S_0(I(jj),:) = W(I(jj),:);
    end

end

DOA_rad = asin(-thet(I)/pi)';
DOA_deg = DOA_rad * (180/pi);

end