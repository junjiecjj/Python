%% TM-CFARµÄÐé¾¯¸ÅÂÊ¼ÆËã
function Pfa = TM_Pfa(alpha,N,rate1,rate2)
    k1 = ceil(N * rate1);
    k2 = ceil(N * rate2);
    
    init = 0;
    for j = 0:k1
        init = init + (N - k1 - k2) * nchoosek(k1,j) * (-1)^(k1-j) /(N - j + alpha);
    end
    vi1 = factorial(N) / (factorial(k1) * factorial(N - k1 - 1) * (N - k1 - k2)) * init;
    
    vi2 = 1;
    for i = 2:N - k1 - k2
        ai = (N - k1 - i + 1) / (N - k1 - k2 - i + 1);
        vi2 = vi2 * ai / (ai + alpha / ( N - k1 - k2));
    end
    
    Pfa = vi1 * vi2;
end
