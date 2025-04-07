clc;clear all;close all;



clc
clear all
close all


% generate signal and array response
fs = 100e3;   % sample rate
f0 = 20e3;    % signal frequency
t = 0:1/fs:0.1;    % time vector
s = sin(2*pi*f0*t); % signal
d = 0.02; % element spacing
N = 8; % number of elements
theta = 10; % target angle
c = 343; % speed of sound
lambda = c/f0; % wavelength
k = 2*pi/lambda; % wavenumber
d_array = d*(0:N-1); % array element positions
phi_array = exp(-1i*k*d_array*cosd(theta)); % array response
x = s'*phi_array; % received signal
X = fftshift(fft2(x)); % 2D FFT
[m, n] = size(X); % size of FFT matrix
theta_x = asind((-m/2:m/2-1)/(m/2)*sin(pi/2)); % angle axis
theta_y = asind((-n/2:n/2-1)/(n/2)*sin(pi/2)); % angle axis
[X_max, I] = max(abs(X(:))); % find max value in FFT matrix
[I_row, I_col] = ind2sub([m n],I); % find row and column index
theta_x(I_row) % estimate angle in x direction
theta_y(I_col) % estimate angle in y direction