function [ CRB ] = CRLB_AOA_Doppler( sig, doa, fd, SNR, b, fc, fs, d )
% CRLB_AOA_Doppler calculate the CRB matrix of angle, Doppler, b and signal

% Y. Sun and Q. Wan, "Position Determination for Moving Transmitter Using
% Single Station," IEEE Access, vol. 6, no. 1, pp. 61103-61116, Oct. 2018.

% input:
%   sig: emitted signal, N x 1
%   doa: DOA, rad, scalar
%   fd: Doppler, scalar
%   SNR: signal-noise-ratio, dB, scalar
%   b: attenuation coefficient, scalar
%   fc: carrier frequency, scalar
%   fs: sampling frequency, scalar
%   d: elements space of array, meter, M x 1
% output:
%   CRB: CRB matrix of angle, Doppler, b and signal

N = length(sig);
M = length(d);
wc = 3e8/fc;

a = exp(1i*2*pi*d*cos(doa)/wc);
e = exp(1i*2*pi*fd*(0:N-1)'/fs);
A = kron(diag(e),a);

D = -1i*2*pi*sin(doa)/wc * kron(eye(N),diag(d));
F = 1i*2*pi/fs * kron(diag(1:N),eye(M));

NsePwr = 10^(-SNR/10);  % signal power is 1
b2 = b'*b;

Jaa = 2/NsePwr*b2 * real((D*A*sig)'*(D*A*sig));
Jaf = 2/NsePwr*b2 * real((D*A*sig)'*F*A*sig);
Jab = 2/NsePwr * [real((b*D*A*sig)'*A*sig),-imag((b*D*A*sig)'*A*sig)];
Jfb = 2/NsePwr * [real((b*F*A*sig)'*A*sig),-imag((b*F*A*sig)'*A*sig)];
Jbb = 2/NsePwr * kron(eye(2),(A*sig)'*(A*sig));

J = [Jaa, Jaf, Jab;
   Jaf',Jff, Jfb;
   Jab',Jfb',Jbb];
CRB = inv(J);

end

