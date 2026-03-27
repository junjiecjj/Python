

clc;
clear all;
close all;


Rho = 0:0.2:1;

N = 12;
Nlst = (-(N/2):N/2-1)';

Theta_deg = (-90:1:90);
Theta = Theta_deg*pi/180;
deltaTheta = Theta(2) - Theta(1);

Ptheta = zeros(length(Rho), length(Theta));
cmap = jet(length(Rho)); % 生成hsv颜色矩阵


%% 
figure(1);
for i = 1:length(Rho)
    rho = Rho(i);
    row = logspace(0, log10(rho^(N-1)), N);
    R = toeplitz(row);
    sum = 0;
    for j = 1:length(Theta)
        theta = Theta(j);
        at = exp(1i * pi * Nlst * sin(theta));
        Ptheta(i,j) = abs(at' * R * at) / (4*pi);
        sum = sum + 2*pi*Ptheta(i,j)*cos(theta)*deltaTheta
    end
    sum; % == N
    plot(Theta_deg, Ptheta(i,:), 'Color', cmap(i,:)); hold on;
end


xlabel('\theta(degree)');
ylabel('P(\theta) (W/ster)');
legend('\rho = 0', '\rho = 0.2', '\rho = 0.4', '\rho = 0.6', '\rho = 0.8', '\rho = 1.0', 'Location', 'northwest');
grid on

%% dB
figure(2);
cmap = jet(length(Rho)); % 生成hsv颜色矩阵
for i = 1:length(Rho)
    rho = Rho(i);
    row = logspace(0, log10(rho^(N-1)), N);
    R = toeplitz(row);
    sum = 0;
    for j = 1:length(Theta)
        theta = Theta(j);
        at = exp(1i * pi * Nlst * sin(theta));
        Ptheta(i,j) = abs(at' * R * at) / (4*pi);
        sum = sum + 2*pi*Ptheta(i,j)*cos(theta)*deltaTheta
    end
    sum; % == N
    Ptheta(i,:) = 10*log10(Ptheta(i,:));
    plot(Theta_deg, Ptheta(i,:), 'Color', cmap(i,:)); hold on;
end


xlabel('\theta(degree)');
ylabel('P(\theta) (dB)');
ylim([-25 25]);
legend('\rho = 0', '\rho = 0.2', '\rho = 0.4', '\rho = 0.6', '\rho = 0.8', '\rho = 1.0', 'Location', 'northwest');
grid on

























