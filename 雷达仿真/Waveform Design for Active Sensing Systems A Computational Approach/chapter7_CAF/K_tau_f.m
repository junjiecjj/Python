function K = K_tau_f(N, tp, tau, f)
% K = K_tau_f(N, tp, tau, f): construct the K matrix for a time delay tau
% and Doppler frequency shift f
%
%   N: number of sub-pulses
%   tp: time duration of each sub-pulse
%   tau: time delay
%   f: Doppler frequency shift
%
%   K: N-by-N

K = zeros(N, N);

v = tau / tp;
if (v <= -N) || (v >= N)
    return;
end

if floor(v) == v % v is an integer
    if v >= 0
        l = v;
        k = 0;
        for n = 1:(N-v)
            l = l + 1; k = k + 1;
            K(l,k) = pulseIntegral(tp, tau, f, l, k);
        end
    else
        l = 0;
        k = -v;
        for n = 1:(N+v)
            l = l + 1; k = k + 1;
            K(l,k) = pulseIntegral(tp, tau, f, l, k);
        end
    end
else % v is not an integer
    v1 = floor(v);
    v2 = ceil(v);
    if v >= 0
        l = v1; k = 0;
        for n = 1:(N-v1)
            l = l + 1; k = k + 1;
            K(l,k) = pulseIntegral(tp, tau, f, l, k);
        end
        l = v2; k = 0;
        for n = 1:(N-v2)
            l = l + 1; k = k + 1;
            K(l,k) = pulseIntegral(tp, tau, f, l, k);
        end
    else
        l = 0; k = -v1;
        for n = 1:(N+v1)
            l = l + 1; k = k + 1;
            K(l,k) = pulseIntegral(tp, tau, f, l, k);
        end
        l = 0; k = -v2;
        for n = 1:(N+v2)
            l = l + 1; k = k + 1;
            K(l,k) = pulseIntegral(tp, tau, f, l, k);
        end
    end
end