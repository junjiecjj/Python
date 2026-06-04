
function CRB = crb_single_63(a, adot, Rs, SNR)
    A = a * a';
    Adot = adot * a' + a * adot';
    term_AA = trace(A * Rs * A');
    term_DD = trace(Adot * Rs * Adot');
    term_DA = trace(Adot * Rs * A');
    CRB = real(term_AA / (2 * SNR * (term_DD * term_AA - abs(term_DA)^2)));
end