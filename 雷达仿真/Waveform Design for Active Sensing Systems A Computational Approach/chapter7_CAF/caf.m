function [CAF t_grid f_grid] = caf(x, y, f_max)
% caf(x, y, f_max): plot the cross ambiguity function of x and y,
% which is defined as chi(tau,f) = \int x(t) y*(t+tau) exp{j*2*pi*f*t}
% dt = (\int y(t) x*(t-tau) exp{-j*2*pi*f*(t-tau)} dt)*
%
%   x: N-by-1, the transmit waveform
%   y (optional): N-by-1, the receive filter
%   f_max (optional): the maximum Doppler shift in the plot, in
%   unit 1/T where T is the total time duration of x
%
%   CAF: (2*Nf-1)-by-(2*Nt-1)
%   t_grid: (2*Nt-1)-by-1
%   f_grid: (2*Nf-1)-by-1

% no. of symbols of x or y
N = length(x);
% time (frequency) normalized w.r.t. T (1/T)
T = 1; 

if nargin <= 2
    % the signal bandwidth is 1/(T/N) = N/T
    f_max = 0.5 * N/T; % half bandwidth
    if nargin == 1
        y = x;
    end
end

% the base sampling interval is T/N for time and 1/T for frequency
sr = 8; % over-sampling rate for the CAF plot
t_grid_size = (T/N) / sr;
Ns = N * sr;
t_grid = (-Ns+1 : Ns-1)' * t_grid_size;

f_grid_size = (1/T) / sr;
Nf = ceil(f_max / f_grid_size);
f_grid = (-Nf+1 : Nf-1)' * f_grid_size;

% over-sample the signal
u = reshape(ones(sr,1) * x.', [Ns 1]); % Ns-by-1
v = reshape(ones(sr,1) * y.', [Ns 1]); % Ns-by-1

% calculate the CAF
CAF = zeros(2*Nf-1, 2*Ns-1);
for p = 1:(2*Nf-1)
    CAF(p,:) = (xcorr(v, u .* exp(1i * 2*pi * f_grid(p) * (0:Ns-1)' * ...
                                  t_grid_size)))';
end

CAF = abs(CAF);
CAF = CAF / max(CAF(:));

% plot 3D CAF (only positive frequency)
t_grid_plot = [-T; t_grid; T];
f_grid_plot = [0; f_grid(Nf:end)];
CAF_plot = [zeros(1,2*Ns-1); CAF(Nf:end,:)];
CAF_plot = [zeros(Nf+1,1) CAF_plot zeros(Nf+1,1)]; % (Nf+1)-by-(2Ns+1)
figure;
mesh(t_grid_plot, f_grid_plot, CAF_plot); hold on;
% zero-Doppler cut surface
surface(t_grid_plot, [0 0], CAF_plot(1:2,:)); hold off;
colormap('default');
axis([-T T 0 f_max 0 1]);
xlabel('$\tau / T$', 'Interpreter', 'LaTex');
ylabel('$f \times T$', 'Interpreter', 'LaTex');
title('$|\chi(\tau,f)|$', 'Interpreter', 'LaTex');
myboldify;

% plot the auto-correlation function
figure;
plot(t_grid, 20*log10(CAF(Nf,:)));
axis([-T T -60 0]);
xlabel('$\tau / T$', 'Interpreter', 'LaTex');
ylabel('$r(\tau / T)$', 'Interpreter', 'LaTex');
myboldify;

% plot the log-scale CAF
figure;
imagesc(t_grid, f_grid, 20*log10(CAF), [-40 0]);
axis([-T T -f_max f_max]);
colormap(flipud(colormap('hot')));
set(gca, 'YDir', 'normal');
xlabel('$\tau / T$', 'Interpreter', 'LaTex');
ylabel('$f \times T$', 'Interpreter', 'LaTex');
title('$|\chi(\tau,f)|$ (dB)', 'Interpreter', 'LaTex');
colorbar;
myboldify;
