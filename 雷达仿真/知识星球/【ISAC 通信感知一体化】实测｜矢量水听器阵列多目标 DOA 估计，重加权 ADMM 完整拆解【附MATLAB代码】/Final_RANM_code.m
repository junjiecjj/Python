clc;
clear;
close all;
rng(1);
% COHERENT NOISE ALONE

%% ===================================================
% 1. SYSTEM PARAMETERS
%% ===================================================
M = 20;             % Number of Sensors
Ns = 300;           % Number of Snapshots
theta_true = [19 156 64 120 250]; % True Target DOAs
theta_grid = 0:1:359;
Ng = length(theta_grid);

% ADMM Parameters
rho = 1.2;          % Penalty parameter (Increased for Alpha-stable stability)
iter = 250;         % More iterations to allow reweighting to settle
lambda_init = 0.6;  % Initial regularization (High to suppress spikes)

%% ===================================================
% 2. CIRCULAR VSA ARRAY + NOD MODEL
%% ===================================================
R = 1.5; % Radius
phi_m = linspace(0,2*pi,M+1);
phi_m(end) = [];

beta_deg = 10; % Non-orthogonal deviation angle
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

% NOD Matrix Construction (Psi)
psi_single = [1 0 0;
              0 1 0;
              0 sin(beta_rad) cos(beta_rad)];
Psi = [];
for m = 1:M
    Psi = blkdiag(Psi, psi_single);
end

% Actual steering matrix under deviation
G = Psi * G_ideal;

%% ===================================================
% 3. SIGNAL GENERATION + INTERFERENCE + NOISE
%% ===================================================
% Target Signals
x_true_signals = zeros(Ng,Ns);
for k = 1:length(theta_true)
    [~,idx] = min(abs(theta_grid - theta_true(k)));
    x_true_signals(idx,:) = (randn(1,Ns) + 1j*randn(1,Ns))/sqrt(2);
end
r = G * x_true_signals;

% External Coherent Interference (e.g., from an Access Point)
theta_interf = [40 140];
s_int = (randn(1,Ns) + 1j*randn(1,Ns))/sqrt(2);
for k = 1:length(theta_interf)
    [~,idx] = min(abs(theta_grid - theta_interf(k)));
    r = r + 0.5 * G(:,idx) * s_int; 
end

% Alpha-Stable (Impulsive) Noise Generation
alpha_val = 1.2; % Characteristic exponent (1.5 = Very impulsive)
gamma = 0.6;     % Dispersion (Scale)
U = pi*(rand(3*M,Ns) - 0.5);
W = -log(rand(3*M,Ns));
noise = gamma * (sin(alpha_val*U) ./ (cos(U)).^(1/alpha_val)) .* ...
        (cos(U - alpha_val*U) ./ W).^((1-alpha_val)/alpha_val);
r = r + noise;

%% ---------- 2. NON-COHERENT INTERFERENCE ----------
theta_interf_nc = [75 260];   % new angles

for k = 1:length(theta_interf_nc)
    [~,idx] = min(abs(theta_grid - theta_interf_nc(k)));
    
    % independent waveform → non-coherent
    s_nc = randn(1,Ns);
    
    r = r + 0.2 * G(:,idx) * s_nc;
end


%% ===================================================
% 4. ROBUST REWEIGHTED ANM-ADMM (FINAL RECALIBRATION)
%% ===================================================
z = zeros(Ng,Ns);
u = zeros(Ng,Ns);
w = ones(Ng,1);      
eps_w = 1e-4;

% --- Pre-processing for Stability ---in
L = diag(ones(Ng,1)) - diag(ones(Ng-1,1),1);
Reg_L = L'*L; 
GtG = G'*G;

% Normalize signal and steering matrix to 0-1 scale
% This is the most stable way to handle Alpha-stable outliers
r_norm = r ./ (abs(r) + 0.2); % Nonlinear Squelch (Pre-filter)
Gtr = G' * r_norm;

% Re-adjust ADMM parameters for normalized space
rho_cal = 0.1; 
lambda_cal = 0.02; 

fprintf('Iterating Robust R-ANM-ADMM...\n');
for it = 1:iter
    % A. PRIMAL UPDATE (LS with Spatial Smoothing)
    % Using Reg_L keeps the NOD (deviation) from causing artifacts
    A = GtG + (rho_cal + 1e-3)*eye(Ng) + 0.02*Reg_L;
    B = Gtr + rho_cal*(z - u); 
    x = A \ B;
    
    % B. ADAPTIVE ANNEALING
    % We slowly lower the threshold to find weaker targets
    current_lambda = lambda_cal * (0.98^it);
current_lambda = max(current_lambda, 0.005);   % lambda_floor
    
    % C. REWEIGHTED SHRINKAGE (The Filter)
    % This is where alpha-stable noise is removed
    z = sign(x + u) .* max(abs(x + u) - (current_lambda * repmat(w, 1, Ns)), 0);
    
    % D. DUAL UPDATE
    u = u + x - z;
    
    % E. WEIGHT UPDATE
    % Updates the spatial filter based on previous iteration
    w = 1 ./ (mean(abs(z), 2) + eps_w);
end
x_est = z;

%% ===================================================
% 5. SPECTRUM & DYNAMIC PEAK DETECTION
%% ===================================================
P = mean(abs(x_est).^2, 2);
if max(P) > 0
    P = P / max(P); 
else
    % Fallback if ADMM was too aggressive
    P = mean(abs(x_est + u).^2, 2);
    P = P / max(P + eps);
end

% Initial Peak Search
threshold = 0.05; % Lowered threshold for sensitivity
min_dist = 8;   
[pk, loc] = findpeaks(P, theta_grid, 'MinPeakHeight', threshold, 'MinPeakDistance', min_dist);

% NOVELTY: Top-K Selection logic
K_expected = length(theta_true); 
if length(loc) > K_expected
    [~, sort_idx] = sort(pk, 'descend');
    loc = loc(sort_idx(1:K_expected));
    loc = sort(loc); 
elseif isempty(loc)
    % Final emergency fallback: detect anything above noise floor
    [pk, loc] = findpeaks(P, theta_grid, 'MinPeakHeight', 0.01);
end
theta_est_final = loc;

%% ===================================================
% 5. SPECTRUM & TOP-K PEAK DETECTION
%% ===================================================
% P = mean(abs(x_est).^2, 2);
% P = P / max(P); % Normalize
% 
% % Initial Peak Search
% threshold = 0.2; 
% min_dist = 12;   % Minimum angular separation
% [pk, loc] = findpeaks(P, theta_grid, 'MinPeakHeight', threshold, 'MinPeakDistance', min_dist);

% NOVELTY: Top-K Selection logic to filter out Alpha-stable "Ghost" peaks
K_expected = length(theta_true); 
if length(loc) > K_expected
    [~, sort_idx] = sort(pk, 'descend');
    loc = loc(sort_idx(1:K_expected));
    loc = sort(loc); % Put back in angular order
end

% ===== ORDER MATCHING WITH TRUE DOA =====
theta_est_ordered = zeros(size(theta_true));
temp_est = loc;   % use detected peaks

for k = 1:length(theta_true)
    
    if isempty(temp_est)
        break
    end
    
    [~, idx] = min(abs(temp_est - theta_true(k)));
    
    theta_est_ordered(k) = temp_est(idx);
    
    temp_est(idx) = [];
end

theta_est_final = theta_est_ordered;

%% ===================================================
% 6. DISPLAY RESULTS & PLOT
%% ===================================================
if isempty(theta_est_final)
    disp('No targets detected.');
else
    % Match for error calculation
    theta_match = [];
    temp = theta_est_final;
    for k = 1:length(theta_true)
        if isempty(temp), break, end
        [~,idx_m] = min(abs(temp - theta_true(k)));
        theta_match(end+1) = temp(idx_m);
        temp(idx_m) = [];
    end
    
    disp('--- Final Results ---')
    disp(['True DOA:      ', num2str(theta_true)])
    disp(['Estimated DOA: ', num2str(theta_est_final)])
    disp(['Detected: ', num2str(length(theta_est_final)), ' / Expected: ', num2str(K_expected)])
end

figure('Color', 'w', 'Position', [100 100 800 600])
subplot(2,1,1)
plot(theta_grid, P, 'LineWidth', 2, 'Color', [0 0.447 0.741])
title('R-ANM-ADMM Spatial Spectrum')
grid on; ylabel('Normalized Power'); xlim([0 360])

subplot(2,1,2)
plot(theta_grid, P, 'LineWidth', 2)
hold on
stem(theta_true, ones(size(theta_true)), 'r--', 'LineWidth', 1.5, 'MarkerFaceColor', 'r')
if ~isempty(theta_est_final)
    stem(theta_est_final, 0.9*ones(size(theta_est_final)), 'g', 'LineWidth', 1.5, 'MarkerFaceColor', 'g')
end
xlabel('Angle (deg)'); ylabel('Spectrum Magnitude')
title('Performance: Multi-Target Detection under Alpha-Stable Noise & NOD')
legend('Recovered Spectrum', 'Ground Truth', 'R-ANM Estimate')
grid on; xlim([0 360])