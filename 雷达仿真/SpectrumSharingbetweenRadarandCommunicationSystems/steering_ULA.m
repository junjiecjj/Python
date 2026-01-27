function a = steering_ULA(theta, M, fc, d)
% Generates a steering vector for Uniform Linear Array (ULA)
% Inputs:
%   theta: angle of arrival with the vertical (in degrees)
%   M: number of antennas
%   fc: carrier frequency (in Hz)
%   d: inter-element spacing (in meter)
% Output:
%   a: Steering vector (M x 1)

c = 3e8; % speed of light in meter/sec
lambda = c / fc; % wavelength in meter
m = 0:M-1;
a = exp(-1i * 2 * pi * m * d * sind(theta - 45) / lambda);
a = a.';
end