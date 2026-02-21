function pseudoRandomSequence = generateMLS(p)
% Generate a pseudorandom sequence of length N=2^p-1, where p is an
% integer. The sequence is generated using shift registers. The feedback
% coefficients for the registers are obtained from the coefficients of an
% irreducible, primitive polynomial in GF(2p).

    primitivePoly = gfprimdf(p, 2);
    sequence = zeros(2^p - 1, 1);
    sequence(1:p) = randi([0 1], p, 1);
    
    for i = (p + 1):(2^p - 1)
        sequence(i) = mod(-primitivePoly(1:p)*sequence(i-p : i-1), 2);
    end
    
    sequence(sequence == 0) = -1;
    pseudoRandomSequence = sequence;
end