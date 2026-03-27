function [CAF t_grid f_grid] = cafper(x, y, f_max)
% cafper(x, y, f_max): plot the cross periodic ambiguity function of two
% periodic waveforms x and y (with period = T)
% chi_T(tau,f) = \int_0^T x(t) y*(t+tau) exp{j*2*pi*f*t} dt
%       = (\int_0^T y(t) x*(t-tau) exp{-j*2*pi*f*(t-tau)} dt)*
%
%   x: N-by-1, the transmit waveform
%   y (optional): N-by-1, the receive filter
%   f_max (optional): the maximum Doppler shift in the plot (with unit 1/T)
%
%   CAF: (2*Nf-1)-by-(2*Nt-1)
%   t_grid: (2*Nt-1)-by-1
%   f_grid: (2*Nf-1)-by-1

N = length(x);
T = 1; % time (frequency) normalized by T (1/T)

if nargin <= 2
    f_max = 0.5 * N/T; % signal bandwidth = 1/(T/N) = N/T
    if nargin == 1
        y = x;
    end
end

% the base sampling interval is T/N for time and 1/T for frequency
sr = 8; % over sampling rate
t_grid_size = (T/N) /sr;
Ns = N * sr;
t_grid = (-Ns+1 : Ns-1)' * t_grid_size;

f_grid_size = (1/T) / sr;
Nf = ceil(f_max / f_grid_size);
f_grid = (-Nf+1 : Nf-1)' * f_grid_size;

% over-sample the signal
u = reshape(ones(sr,1) * x.', [Ns 1]); % Ns-by-1
v = reshape(ones(sr,1) * y.', [Ns,1]); % Ns-by-1

% calculate the CAF
CAF = zeros(2*Nf-1, 2*Ns-1);
for p = 1:(2*Nf-1)    
    uReturn = [u; u; u] .* exp(1i * 2*pi * f_grid(p) * ...
        t_grid_size * (-Ns:2*Ns-1)');
    % uReturn(Ns+1) corresponds to the zero-time index
    for k = (-Ns+1 : Ns-1)
        t0shift = (Ns+1) - k;
        CAF(p, k+Ns) = uReturn(t0shift : t0shift+Ns-1)' * v;
    end
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
xlabel('\tau / T');
ylabel('f T');
myboldify;

% % plot the auto-correlation function
% figure;
% plot(t_grid, 20*log10(CAF(Nf,:)));
% axis([-T T -60 0]);
% xlabel('\tau / T');
% ylabel('r(\tau / T)');
% myboldify;

% plot the log-scale CAF
figure;
imagesc(t_grid, f_grid, 20*log10(CAF), [-40 0]);
axis([-T T -f_max f_max]);
colormap(flipud(colormap('hot')));
set(gca, 'YDir', 'normal');
xlabel('\tau / T');
ylabel('f T');
colorbar;
myboldify;