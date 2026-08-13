function s = func_bpsk_mod(b)
% FUNC_BPSK_MOD  Manual binary-PSK modulation (drop-in for pskmod(b,2)).
%   s = FUNC_BPSK_MOD(b) maps bits b in {0,1} to BPSK symbols s in {+1,-1}
%   using MATLAB's default convention s = 1 - 2*b  (0 -> +1, 1 -> -1),
%   i.e. exp(1j*pi*b) with zero phase offset. Shape of b is preserved.
%   Matches pskmod(b,2). See manuscript Sec. 5.3, s_{k,i} = 1 - 2 b_{k,i}.
    s = complex(1 - 2 * double(b));
end
