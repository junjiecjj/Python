function avpsl = helperAveragePSL(X)
    [n, m] = size(X);
    psl = zeros(n, 1);

    for i = 1:n
        acmag = ambgfun(X(i, :), 1, 1/m, "Cut", "Doppler");
        psl(i) = acmag(m)/max([acmag(1:m-1) acmag(m+1:end)]);
    end

    avpsl = mag2db(mean(psl));
end