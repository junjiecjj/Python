function seq = helperMLS(p)
% Generate a pseudorandom sequence of length N=2^p-1, where p is an
% integer. The sequence is generated using shift registers. The feedback
% coefficients for the registers are obtained from the coefficients of an
% irreducible, primitive polynomial in GF(2p).

    pol = gfprimdf(p, 2);
    seq = zeros(2^p - 1, 1);
    seq(1:p) = randi([0 1], p, 1);
    
    for i = (p + 1):(2^p - 1)
        seq(i) = mod(-pol(1:p)*seq(i-p : i-1), 2);
    end
    
    seq(seq == 0) = -1;
end