function S = zc_beamformed(Nt, Nc, Ns, a_tx)
% ZC_BEAMFORMED  Generate beamformed Zadoff-Chu sequences.
%
% Inputs:
%   Nt   : Number of transmit antennas.
%   Nc   : Number of subcarriers.
%   Ns   : Number of OFDM symbols.
%   a_tx : Transmit steering vector.
%
% Output:
%   S    : Beamformed radar waveform of size Nt x Nc x Ns.

all_roots = 1:2:Nc-1;
u_list = all_roots(1:Ns);
n = (0:Nc-1).';
S = zeros(Nt, Nc, Ns);
for l = 1:Ns
    u = u_list(l);
    c = exp(-1j*pi*u*(n.^2)/Nc); 
    S(:,:,l) = a_tx * c.';
end
end