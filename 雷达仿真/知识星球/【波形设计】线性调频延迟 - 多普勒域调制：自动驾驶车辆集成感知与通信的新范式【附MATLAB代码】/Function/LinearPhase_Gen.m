function q = LinearPhase_Gen(Nt, N_c)

index = 0:N_c/Nt:N_c-1;
q = exp(1j*2*pi/N_c*index.'*(0:N_c-1));
end