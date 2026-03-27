function rho = par(x)
% rho = par(x), the peak-average-power-ratio

rho = (max(abs(x)))^2 / ((norm(x))^2 / length(x));