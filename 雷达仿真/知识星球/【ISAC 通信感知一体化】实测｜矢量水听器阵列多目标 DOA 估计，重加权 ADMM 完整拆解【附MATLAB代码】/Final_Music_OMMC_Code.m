clc;
clear;
close all;
rng(1);

%% ===================================================
% 1. SYSTEM PARAMETERS
%% ===================================================
M = 20;
Ns = 300;
theta_true = [19 156 64 120 250];

theta_grid = 0:1:359;
Ng = length(theta_grid);

%% ===================================================
% 2. CIRCULAR VSA ARRAY + NOD MODEL
%% ===================================================
R = 1.5;
phi_m = linspace(0,2*pi,M+1);
phi_m(end) = [];

beta_deg = 10;
beta_rad = beta_deg*pi/180;

G_ideal = zeros(3*M,Ng);

for i = 1:Ng
    ang = theta_grid(i)*pi/180;
    for m = 1:M
        ap = exp(1j*2*pi*R*cos(ang - phi_m(m)));
        G_ideal(m,i)       = ap;
        G_ideal(M+m,i)     = ap*cos(ang);
        G_ideal(2*M+m,i)   = ap*sin(ang);
    end
end

psi_single = [1 0 0;
              0 1 0;
              0 sin(beta_rad) cos(beta_rad)];

Psi = [];
for m = 1:M
    Psi = blkdiag(Psi, psi_single);
end

G = Psi * G_ideal;

%% ===================================================
% 3. SIGNAL + INTERFERENCE + SAME NOISE AS RANM
%% ===================================================
x_true_signals = zeros(Ng,Ns);

for k = 1:length(theta_true)
    [~,idx] = min(abs(theta_grid - theta_true(k)));
    x_true_signals(idx,:) = (randn(1,Ns) + 1j*randn(1,Ns))/sqrt(2);
end

r = G * x_true_signals;

% Coherent interference
theta_interf = [40 140];
s_int = (randn(1,Ns) + 1j*randn(1,Ns))/sqrt(2);

for k = 1:length(theta_interf)
    [~,idx] = min(abs(theta_grid - theta_interf(k)));
    r = r + 0.5 * G(:,idx) * s_int;
end

% Alpha-stable noise (SAME AS RANM)
alpha_val = 1.2;
gamma = 0.6;

U = pi*(rand(3*M,Ns) - 0.5);
W = -log(rand(3*M,Ns));

noise = gamma * (sin(alpha_val*U) ./ (cos(U)).^(1/alpha_val)) .* ...
        (cos(U - alpha_val*U) ./ W).^((1-alpha_val)/alpha_val);

r = r + noise;

% Non-coherent interference
theta_interf_nc = [75 260];

for k = 1:length(theta_interf_nc)
    [~,idx] = min(abs(theta_grid - theta_interf_nc(k)));
    s_nc = randn(1,Ns);
    r = r + 0.2 * G(:,idx) * s_nc;
end

%% ===================================================
% 4. MUSIC-OMMC (ONLY ALGORITHM CHANGE)
%% ===================================================
Rxx = (r * r') / Ns;

% OMMC diagonal loading
delta = 0.01 * trace(Rxx)/(3*M);
Rxx = Rxx + delta * eye(3*M);

[U,S,~] = svd(Rxx);

K = length(theta_true);
En = U(:, K+1:end);

%% ===================================================
% 5. MUSIC SPECTRUM
%% ===================================================
P = zeros(Ng,1);

for i = 1:Ng
    a = G(:,i);
    P(i) = 1 / (a' * (En*En') * a);
end

P = abs(P);
P = P / max(P);

%% ===================================================
% 6. PEAK DETECTION (MATCHED WITH RANM)
%% ===================================================
threshold = 0.05;
min_dist = 8;

[pk, loc] = findpeaks(P, theta_grid, ...
    'MinPeakHeight', threshold, 'MinPeakDistance', min_dist);

K_expected = length(theta_true);

if length(loc) > K_expected
    [~, idx] = sort(pk,'descend');
    loc = loc(idx(1:K_expected));
    loc = sort(loc);
elseif isempty(loc)
    [pk, loc] = findpeaks(P, theta_grid, 'MinPeakHeight', 0.01);
end

%% ===================================================
% ORDER MATCHING
%% ===================================================
theta_est_ordered = zeros(size(theta_true));
temp_est = loc;

for k = 1:length(theta_true)
    if isempty(temp_est), break, end
    
    [~, idx] = min(abs(temp_est - theta_true(k)));
    theta_est_ordered(k) = temp_est(idx);
    temp_est(idx) = [];
end

theta_est_final = theta_est_ordered;

%% ===================================================
% DISPLAY
%% ===================================================
disp('--- RESULTS ---')
disp(['True DOA:      ', num2str(theta_true)])
disp(['Estimated DOA: ', num2str(theta_est_final)])

%% ===================================================
% PLOTS 
%% ===================================================
figure('Color', 'w', 'Position', [100 100 900 600])

%% ---------- SUBPLOT 1: Spectrum Only ----------
subplot(2,1,1)
plot(theta_grid, P, 'LineWidth', 2)
title('MUSIC-OMMC Spatial Spectrum')
ylabel('Normalized Power')
grid on
xlim([0 360])

%% ---------- SUBPLOT 2: Spectrum + True + Estimated ----------
subplot(2,1,2)
plot(theta_grid, P, 'LineWidth', 2)
hold on

% True DOA (red dashed with markers)
stem(theta_true, ones(size(theta_true)), ...
    'r--', 'LineWidth', 1.5, 'MarkerFaceColor', 'r')

% Estimated DOA (green)
if ~isempty(theta_est_final)
    stem(theta_est_final, 0.9*ones(size(theta_est_final)), ...
        'g', 'LineWidth', 1.5, 'MarkerFaceColor', 'g')
end

xlabel('Angle (deg)')
ylabel('Spectrum Magnitude')
title('Performance: MUSIC-OMMC under Alpha-Stable Noise & NOD')

legend('Recovered Spectrum', 'Ground Truth', 'MUSIC Estimate')

grid on
xlim([0 360])