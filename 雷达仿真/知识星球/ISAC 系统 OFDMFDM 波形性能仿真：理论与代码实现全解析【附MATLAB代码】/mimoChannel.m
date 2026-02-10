function H = mimoChannel(Nr, Nt)
H = (randn(Nr,Nt) + 1j*randn(Nr,Nt)) / sqrt(2);
end
