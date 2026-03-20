%%
function v = MMSE_QPSK(snr)

max=100;
v = zeros(size(snr));
for i=1:length(snr)
    v(i) = 1 - quad(@(x) f_QPSK(x,snr(i)), -max, max);
end

%% 
function y = f_QPSK(x,snr) 
y  = exp(-x.^2/2)/sqrt(2*pi) .* tanh(snr - sqrt(snr).*x);
end

end