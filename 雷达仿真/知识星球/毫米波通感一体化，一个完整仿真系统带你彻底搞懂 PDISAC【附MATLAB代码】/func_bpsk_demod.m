function b = func_bpsk_demod(y)
% FUNC_BPSK_DEMOD  Manual binary-PSK demodulation (drop-in for pskdemod(y,2)).
%   b = FUNC_BPSK_DEMOD(y) performs the hard decision at the imaginary axis:
%   Re{y} >= 0 -> 0, Re{y} < 0 -> 1. Shape of y is preserved. Matches
%   pskdemod(y,2) with zero phase offset. See manuscript Sec. 5.3.
    b = double(real(y) < 0);
end
