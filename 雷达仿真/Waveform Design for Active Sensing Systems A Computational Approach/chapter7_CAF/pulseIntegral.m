function value = pulseIntegral(tp, tau, f, l, k)
% value = pulseIntegral(tp, tau, f, l, k): calculate the CAF of two
% rectangular pulses. \int p_l(t) p*_k(t + tau) exp(j 2pi ft) dt where
% p_k(t) = 1/sqrt(tp) rect((t - (k-1)tp) / tp)

v = tau / tp;
if ((l-k) >= v) && ((l-k) < (v+1))
    if f ~= 0
        value = (exp(1i * 2*pi * f * k * tp) - ...
            exp(1i * 2*pi * f * (l-1) * tp) * exp(-1i * 2*pi * f * tau)) / ...
            (1i * 2*pi * f * tp);
    else
        value = 1 - (l - k - tau/tp);
    end
elseif ((l-k) > (v-1)) && ((l-k) <= v)
    if f ~= 0
        value = (exp(1i * 2*pi * f * l * tp) * exp(-1i * 2*pi * f * tau) - ...
            exp(1i * 2*pi * f * (k-1) * tp)) / (1i * 2*pi * f * tp);
    else
        value = (l - k - tau/tp) + 1;
    end
else
    value = 0;
end