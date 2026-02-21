clc;
clear;
close all;
tic;
%% Parameters
MonteCarlo = 1000;
M = 2;                  % Number of sources
Mx = 6; My = 6;         % URA: Mx rows, My columns
L = Mx * My;            % Total number of elements
d = 0.5;                % Element spacing (in lambda units)
SNR_dB = 10;            % SNR in dB
SNR = 10^(SNR_dB/10);   % Linear SNR
lambda = 1;
K = 200;                % Snapshots
sigma2 = 1/SNR;

% True DOAs
az_true = [-40, 45];    % degrees
el_true = [30, 40];     % degrees

% Scanning grid
az_scan = -90:1:90;
el_scan = 0:1:90;

% Preallocation
est_az_all = zeros(MonteCarlo, M);
est_el_all = zeros(MonteCarlo, M);

% Sensor positions
[mx, my] = meshgrid(0:Mx-1, 0:My-1);
mx = mx(:); my = my(:);
sum=zeros(length(el_scan), length(az_scan));
for trial = 1:MonteCarlo
    %% Generate array manifold matrix A
    A = zeros(L, M);
    for k = 1:M
        az = deg2rad(az_true(k));
        el = deg2rad(el_true(k));
        ux = sin(el) * cos(az);
        uy = sin(el) * sin(az);
        A(:,k) = exp(1j * 2*pi*d * (mx*ux + my*uy));
    end

    %% Generate signal and noise
    S = (randn(M, K) + 1j*randn(M, K)) / sqrt(2);
    W = sqrt(sigma2) * (randn(L, K) + 1j*randn(L, K)) / sqrt(2);
    X = A * S + W;

    %% Nyström Approximation
    Na = 6;
    indices = randperm(L, Na);
    Y = X(indices, :);
    A_y = A(indices, :);

    R_yy = (Y * Y') / K;
    R_xy = (X * Y') / K;

    [U_y, D_y] = eig(R_yy);
    [~, idx] = sort(diag(D_y), 'descend');
    U_y = U_y(:, idx);
    D_y = D_y(idx, idx);

    lambda_y = D_y(1:M, 1:M);
    U_ns = zeros(L, M);
    for i = 1:M
        U_ns(:,i) = (1 / lambda_y(i,i)) * R_xy * U_y(:, i);
    end

    %% MUSIC Spectrum Calculation (2D)
    Pmusic = zeros(length(el_scan), length(az_scan));
    for i = 1:length(el_scan)
        for j = 1:length(az_scan)
            az = deg2rad(az_scan(j));
            el = deg2rad(el_scan(i));
            ux = sin(el) * cos(az);
            uy = sin(el) * sin(az);
            a = exp(1j * 2*pi*d * (mx*ux + my*uy));
            Pmusic(i,j) = real(1 / (a' * (eye(L) - U_ns * U_ns') * a));
           
        end
    end
     sum=sum+Pmusic;
     Pmusic=sum;
end
% Normalize spectrum
Pmusic_dB = 10 * log10(abs(Pmusic) / max(abs(Pmusic(:))));

% Detect local maxima
BW = imregionalmax(Pmusic);
[rowMax, colMax] = find(BW);
numPeaks = length(rowMax);

% Collect (azimuth, elevation) coords of peaks
peak_coords = zeros(numPeaks, 2);
for k = 1:numPeaks
    peak_coords(k, :) = [az_scan(colMax(k)), el_scan(rowMax(k))];
end

% Cluster the peaks into M=2 groups
if numPeaks >= M
    [cluster_labels, cluster_centers] = kmeans(peak_coords, M, 'Replicates', 5);

    % Average coordinates of each cluster
    est_az = zeros(1, M);
    est_el = zeros(1, M);
    for i = 1:M
        cluster_points = peak_coords(cluster_labels == i, :);
        est_az(i) = mean(cluster_points(:,1));
        est_el(i) = mean(cluster_points(:,2));
    end
else
    % Fallback: not enough local peaks, use maxk
    [~, fallback_idx] = maxk(Pmusic(:), M);
    [r, c] = ind2sub(size(Pmusic), fallback_idx);
    est_az = az_scan(c);
    est_el = el_scan(r);
end
figure;
contour(az_scan, el_scan, Pmusic_dB, 30);
xlabel('Azimuth (degrees)');
ylabel('Elevation (degrees)');
title('2D MUSIC Spectrum (Contour)');
colorbar;
hold on;
plot(az_true, el_true, 'rx', 'MarkerSize', 10, 'LineWidth', 2); % True DOAs
plot(est_az, est_el, 'bo', 'MarkerSize', 10, 'LineWidth', 2); % Estimated DOAs
legend('Spectrum', 'True DOAs', 'Estimated DOAs');
grid on;
% Sort the estimates
est_az_all(trial, :) = sort(est_az);
est_el_all(trial, :) = sort(est_el);
disp("Estimated Azimuth Angles:");
disp(est_az_all(MonteCarlo,:));

disp("Estimated Elevation Angles:");
disp(est_el_all(MonteCarlo,:));
toc;
