function P_music = music_spectrum(y_nl, Ms_est, theta_range)
% Compute the MUSIC spatial spectrum from the received data.
%
% Inputs:
%   y_nl   : Received signal tensor of size Nr x N x L
%   Ms_est : Estimated signal subspace dimension
%   N_grid : Number of angular grid points
%
% Output:
%   P_music : MUSIC spectrum over the angular grid
[Nr, N, L] = size(y_nl);
b_n = @(theta) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)') ./ sqrt(Nr);

% Sample covariance matrix
Y_theta = reshape(y_nl, Nr, []);
R_sample = (Y_theta * Y_theta') / L /N;

% Eigen-decomposition and noise subspace extraction
[U, D] = eig(R_sample);
[~, idx] = sort(real(diag(D)), 'descend');
U = U(:, idx);
Un = U(:, Ms_est+1:end);

% Evaluate the MUSIC spectrum over the angular grid
P_music = zeros(size(theta_range));
for i = 1:length(theta_range)
    b_theta = b_n(theta_range(i));
    P_music(i) = 1 / (b_theta' * (Un * Un') * b_theta);
end
P_music = real(P_music);
P_music = max(P_music, 0);
end