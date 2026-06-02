



%% Local Function
function Pd = radar_detection_probability_fig4(X, thetaDetect, snr, Pfa)
    [M, L] = size(X);
    a = exp(1j * pi * (0:M-1)' * sind(thetaDetect));
    Rs = X * X' / L;
    noncentrality = snr * abs(a' * Rs.' * a)^2;
    delta = chi2inv(1 - Pfa, 2);
    Pd = 1 - ncx2cdf(delta, 2, noncentrality);
end