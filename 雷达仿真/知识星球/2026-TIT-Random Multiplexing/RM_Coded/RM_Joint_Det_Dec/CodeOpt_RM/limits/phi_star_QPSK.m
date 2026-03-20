function v = phi_star_QPSK(x,snr,beta)
v = min(beta^-1*(x.^-1-snr^-1), MMSE_QPSK(x));
end