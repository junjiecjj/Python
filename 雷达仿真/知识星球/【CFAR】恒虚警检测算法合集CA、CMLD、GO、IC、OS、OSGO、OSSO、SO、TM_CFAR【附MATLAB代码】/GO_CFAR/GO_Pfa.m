%% GO-CFARµÄÐé¾¯¸ÅÂÊ¼ÆËã
function Pfa = GO_Pfa(alpha,N)
    Pfa = 0;
    n = ceil(N / 2);
    for i = 0:n-1
        Pfa = Pfa - 2 * gamma(n+i) ./ gamma(i+1) ./ gamma(n) .* (2 + alpha ./ n) .^ (-(n+i));
    end
    Pfa = Pfa + 2 .* (1 + alpha ./ n) .^ (-n);
end
