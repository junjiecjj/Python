function q = PS_Gen(Nt, N_c)

% q = exp(1j*2*pi*rand(Nt, N_c));
q = exp(1j*pi*randi([0 1], Nt, N_c));

end