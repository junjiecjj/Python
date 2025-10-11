%% CMLD-CFARµÄÐé¾¯¸ÅÂÊ¼ÆËã
function Pfa = CMLD_Pfa(alpha,N,rate)
    k = ceil(N * rate);
    Pfa = 0;
    for i = 1:N-k
        ai = nchoosek(N,k) * nchoosek(N-k,i-1) * (-1)^(i-1) * ((N-i+1-k)/k)^(N-k-1);
        ci = (N-i+1) / (N-k-i+1);
        Pfa = Pfa + ai / (ci + alpha);
    end
end