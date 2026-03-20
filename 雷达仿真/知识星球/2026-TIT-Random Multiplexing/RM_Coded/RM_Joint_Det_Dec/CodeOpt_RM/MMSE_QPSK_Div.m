function [v] = MMSE_QPSK_Div(snr)
max=100;
v = zeros(size(snr));
for i=1:length(snr)
    v(i) = 1 - integral(@(x) f_QPSK(x,snr(i)), -max, max);
end

%% 
function y = f_QPSK(x,snr) 
y  = exp(-x.^2/2)/sqrt(2*pi) .* tanh(snr - sqrt(snr).*x);
end

end

