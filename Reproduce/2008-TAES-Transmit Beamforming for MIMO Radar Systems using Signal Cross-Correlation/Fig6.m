

clc;
clear all;
close all;
addpath('./functions');


rng('default');                     % Set random number generator for reproducibility


N = 10;
d = 0.5;
lambda = 2*d;

% Use isotropic antenna elements
element = phased.IsotropicAntennaElement('BackBaffled', true);
array = phased.ULA('Element', element, 'NumElements', N, 'ElementSpacing', d, 'ArrayAxis', 'y');


% Three targets of interest
tgtAz = [0];                % Azimuths of the targets of interest

ang = linspace(-90, 90, 200);       % Grid of azimuth angles
beamwidth = 36;                     % Desired beamwidth

% Desired beam pattern
idx = false(size(ang));
for i = 1:numel(tgtAz)
    idx = idx | ang >= tgtAz(i)-beamwidth/2 & ang <= tgtAz(i)+beamwidth/2;
end

Bdes = zeros(size(ang));
Bdes(idx) = 1;

figure(1);
plot(ang, Bdes, 'LineWidth', 2);
xlabel('Azimuth (deg)');
ylabel('Desired Beam Pattern');
title('Desired Beam Pattern');
grid on;
ylim([0 1.1]);


%%  A. Squared Error Optimization
Pt = 1;
rxpos = array.getElementPosition(); 

% Normalize the antenna element positions by the wavelength
normalizedPos = rxpos/lambda;

% Solve the optimization problem to find the covariance matrix
Rmmse = helperMMSECovariance(normalizedPos, Bdes, ang); % Eq.(24)

Rmmse = Rmmse*Pt/N;

% Matrix of steering vectors corresponding to the angles in the grid ang
A = steervec(normalizedPos, [ang; zeros(size(ang))]);

% Compute the resulting beam pattern given the found covariance matrix
Bmmse = abs(diag(A'*Rmmse*A))/(4*pi);
% 
% figure(2);
% hold on
% plot(ang, pow2db(Bdes + eps), 'LineWidth', 2)
% plot(ang, pow2db(Bmmse/max(Bmmse)), 'LineWidth', 2)
% 
% grid on
% xlabel('Azimuth (deg)')
% ylabel('(dB)');
% legend('Desired', 'MMSE Covariance');
% ylim([-40 1]);
% title('Transmit Beam Pattern');


%%  B. Maximum Error Optimization
% Solve the optimization problem to find the covariance matrix
Rminmax = helperMinMaxCovariance(normalizedPos, Bdes, ang);

Rminmax = Rminmax*Pt/N;

% Matrix of steering vectors corresponding to the angles in the grid ang
A = steervec(normalizedPos, [ang; zeros(size(ang))]);

% Compute the resulting beam pattern given the found covariance matrix
Bminmax = abs(diag(A'*Rminmax*A))/(4*pi);

figure(1);
plot(ang, pow2db(Bdes + eps), 'LineStyle','-', 'LineWidth', 2, 'Color','k'); hold on;
plot(ang, pow2db(Bmmse/max(Bmmse)), 'LineStyle','-', 'LineWidth', 2, 'Color','r'); hold on;
plot(ang, pow2db(Bminmax/max(Bminmax)), 'LineStyle','-', 'LineWidth', 2, 'Color','b');


xlabel('Azimuth (deg)');
ylabel('(dB)');
legend('Desired', 'MMSE Covariance', 'MinMax Covariance');
ylim([-40 1]);
title('Transmit Beam Pattern');
grid on;

























































































