function y = PSF_RaiCos(tau, alpha, Ts, tau_pulse)
    y = zeros(size(tau));
    idx = find(abs(tau)<=tau_pulse);
    x = tau(idx)./Ts;
    Num = cos(alpha*pi*x);
    Den = (1-(2*alpha*x).^2);
    Result = Num ./ Den;
    Result(abs(Den)<eps) = pi/4;
    y(idx) = sinc(x) .* Result;
end