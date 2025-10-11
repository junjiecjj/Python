%% OSSO-CFARµÄÐé¾¯¸ÅÂÊ¼ÆËã
function Pfa = OSSO_Pfa(alpha,N,rate)
    k = ceil(N * rate / 2);
    pfa = 0;
    for j = 0:N/2-k
        for i = 0:N/2-k
            pfa = pfa + nchoosek(N/2-k,j)*nchoosek(N/2-k,i)*((-1)^(N-2*k-j-i)*gamma(N-j-i)*gamma(alpha+1))/((N/2-i)*gamma(N-j-i+alpha+1));
        end
    end
    
    Pfa = 2 * k * nchoosek(N/2,k) * ((gamma(k) * gamma(alpha+N/2-k+1)/gamma(alpha+N/2+1)) - k * nchoosek(N/2,k) * pfa);
end