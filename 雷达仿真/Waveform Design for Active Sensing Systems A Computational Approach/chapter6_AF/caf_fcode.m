function [CAF t_grid f_grid] = caf_fcode(x, x_fcode, y, y_fcode, f_max, T)
% caf_fcode(x, x_fcode, y, y_fcode, f_max, T): plot the cross ambiguity 
% function of x and y, with frequency code x_fcode and y_fcode
% NOTE: f_max and T are not normalized values (different from caf)
%
%   x: N-by-1, the transmit waveform
%   x_fcode: N-by-1
%   y: N-by-1, the receive filter
%   y_fcode: N-by-1
%   f_max: the maximum Doppler shift in the plot
%   T: the total time duration of the waveform
%
%   CAF: (2*Nf-1)-by-(2*Nt-1)
%   t_grid: (2*Nt-1)-by-1
%   f_grid: (2*Nf-1)-by-1

% no. of symbols of x or y
N = length(x);

% the base sampling is T/N for time and 1/T for frequency
sr = 10; % over-sampling rate for the CAF plot
t_grid_size = (T/N) / sr;
Ns = N * sr;
t_grid = (-Ns+1 : Ns-1)' * t_grid_size;

f_grid_size = (1/T) / sr;
Nf = ceil(f_max / f_grid_size);
f_grid = (-Nf+1 : Nf-1)' * f_grid_size;

% over-sample the signal
u = reshape(ones(sr,1) * x.', [Ns 1]); % Ns-by-1
v = reshape(ones(sr,1) * y.', [Ns 1]); % Ns-by-1

% frequency cumulation
freq_u = reshape(ones(sr,1) * x_fcode.', [Ns 1]); % Ns-by-1
freq_v = reshape(ones(sr,1) * y_fcode.', [Ns 1]); % Ns-by-1
u = u .* exp(1i * 2*pi * cumsum(freq_u * t_grid_size));
v = v .* exp(1i * 2*pi * cumsum(freq_v * t_grid_size));

% calculate the CAF
CAF = zeros(2*Nf-1, 2*Ns-1);
for p = 1:(2*Nf-1)
    CAF(p,:) = (xcorr(v, u .* exp(1i * 2*pi * f_grid(p) * (0:Ns-1)'* ...
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
mesh(t_grid_plot/T, f_grid_plot*T, CAF_plot); hold on;
% zero-Doppler cut surface
surface(t_grid_plot/T, [0 0], CAF_plot(1:2,:)); hold off;
colormap('default');
axis([-1 1 0 f_max*T 0 1]);
xlabel('\tau / T');
ylabel('f T');
myboldify;

% % plot the auto-correlation function
% figure;
% plot(t_grid/T, 20*log10(CAF(Nf,:)));
% axis([-1 1 -60 0]);
% xlabel('\tau / T');
% ylabel('f T');
% myboldify;
% 
% plot the log-scale CAF
figure;
imagesc(t_grid/T, f_grid*T, 20*log10(CAF), [-40 0]);
axis([-1 1 -f_max*T f_max*T]);
colormap(flipud(colormap('hot')));
set(gca, 'YDir', 'normal');
xlabel('\tau / T');
ylabel('f T');
colorbar;
myboldify;
