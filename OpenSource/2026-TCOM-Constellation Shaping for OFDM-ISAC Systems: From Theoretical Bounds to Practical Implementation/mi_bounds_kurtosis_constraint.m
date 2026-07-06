% =========================================================================
% Constellation Shaping for OFDM-ISAC Systems:
% From Theoretical Bounds to Practical Implementation
%
% Author: Benedikt Geiger
%
% This MATLAB script numerically evaluates upper and lower bounds on the
% achievable mutual information under power and kurtosis constraints,
% using maximum-entropy input distributions.
%
% The implementation corresponds to Eqs. (23)–(32) in:
% [1] B. Geiger, F. Liu, S. Lu, A. Rode, D. Gil Gaviria, C. Muth, and
%     L. Schmalen, "Constellation shaping for OFDM-ISAC systems: From
%     theoretical bounds to practical implementation",
%     submitted to IEEE Transactions on Communications, 2025.
%     Available at: https://arxiv.org/abs/2509.04055
%
% License: MIT (see LICENSE in repository root)
% =========================================================================

clear; close all; clc;

%% Define parameter for the optimisation
SNR_dB = 10;                                                % Comms SNR in dB
sigma2_c = 10^(-SNR_dB/10);                                 % Noise power (Comms) of AWGN
entropy_AWGN = log2(pi*sigma2_c*exp(1));                    % Entropy of the AWGN
E_s = 1;                                                    % Input power
capacity_AWGN_channel = log2(1+E_s/sigma2_c);               % Capacity of an AWGN channel (reference)
kurtosis_array = 1.00:0.01:1.99;                            % Array of kurtosis constraints

C_0x = 1;                                                   % Probability constraint input
C_0y = 1;                                                   % Probability constraint output
C_1x = E_s;                                                 % Power constraint input (assumed to be 1)
C_1y = E_s + sigma2_c;                                      % Power constraint output
C_2x = kurtosis_array.';                                    % Kurtosis constraint input
C_2y = kurtosis_array.' + 4*E_s*sigma2_c + 2*sigma2_c^2;    % Kurtosis constraint output

% Sweep through the kurtosis values
gamma_results_input = zeros(length(kurtosis_array), 3);     % gamma_0, gamma_2, and gamma_4 are in the first, second, third column respectively
gamma_results_output = zeros(length(kurtosis_array), 3);
for i = 1:length(kurtosis_array)
    % Single nonlinear equation (29)
    nonlinear_equation_input = @(gamma2) ((C_1x*(gamma2*C_1x+1)/C_2x(i) - gamma2)*sqrt(pi*C_2x(i)/(2*(gamma2*C_1x+1)))*exp(gamma2^2*C_2x(i)/(2*(gamma2*C_1x+1)))*erfc(-gamma2*sqrt(C_2x(i)/(2*(gamma2*C_1x+1))))-C_0x);
    nonlinear_equation_output = @(gamma2) ((C_1y*(gamma2*C_1y+1)/C_2y(i) - gamma2)*sqrt(pi*C_2y(i)/(2*(gamma2*C_1y+1)))*exp(gamma2^2*C_2y(i)/(2*(gamma2*C_1y+1)))*erfc(-gamma2*sqrt(C_2y(i)/(2*(gamma2*C_1y+1))))-C_0y);

    try
        % Try to solve the nonlinear equation (29)
        gamma_results_input(i,2) = fzero(nonlinear_equation_input, [-1/E_s+1e-6, 300]);
    catch
        % If numerical instabilities occur, solve the system of equations
        % (28) instead
        gamma_results_input(i,:) = fsolve(@(gammas) system_of_equations(gammas, C_0x, C_1x, C_2x(i)),[-1*C_1x,-0.5*C_1x,-0.01*C_1x]);
    end

    try
        % Try to solve the nonlinear equation (29)
        gamma_results_output(i,2) = fzero(nonlinear_equation_output, [-1/(E_s+sigma2_c)+5e-3, 150]);
    catch
        % If numerical instabilities occur, solve the system of equations
        % (28) instead
        gamma_results_output(i,:) = fsolve(@(gammas) system_of_equations(gammas, C_0y, C_1y, C_2y(i)),[-1*C_1y,-0.5*C_1y,-0.01*C_1y]);
    end
end

%% Calculate gamma0 (30) and gamma4 (31)
% Input
gamma_results_input(:,1) = log(1/pi*(C_1x.*(gamma_results_input(:,2).*C_1x+1)./C_2x-gamma_results_input(:,2)));
gamma_results_input(:,3) = -1./(2*C_2x).*(gamma_results_input(:,2)*C_1x+1);
% Output
gamma_results_output(:,1) = log(1/pi*(C_1y.*(gamma_results_output(:,2).*C_1y+1)./C_2y-gamma_results_output(:,2)));
gamma_results_output(:,3) = -1./(2*C_2y).*(gamma_results_output(:,2)*C_1y+1);


%% Calculate the entropy (32)
entropy_input = -1/log(2)*(gamma_results_input(:,1)+C_1x.*gamma_results_input(:,2)+C_2x.*gamma_results_input(:,3));
entropy_output = -1/log(2)*(gamma_results_output(:,1)+C_1y.*gamma_results_output(:,2)+C_2y.*gamma_results_output(:,3));


%% Calculate the bounds (23)/(25)
lower_bound = log2(2.^entropy_input + 2.^entropy_AWGN) - entropy_AWGN;
upper_bound = entropy_output - entropy_AWGN;


%% Plot results
figure();
yline(capacity_AWGN_channel, "Linewidth", 1, "Color","black");
hold on;
plot(kurtosis_array, upper_bound, "LineWidth",3);
plot(kurtosis_array, lower_bound, "LineWidth",3);
legend(["AWGN Channel Capacity", "Upper Bound", "Lower Bound"], 'Location','southeast');
xlabel("Kurtosis constraint \kappa");
ylabel("Rate (bit/Symbol)");
title(sprintf('SNR = %d dB', SNR_dB));
grid on;

%% Function definitions for system of equations (fallback solution)
% Define function under the integral
function integrand = integrand_moment0(x_real, x_imag, gamma0, gamma2, gamma4)
    integrand = exp(gamma0 + gamma2*(x_real.^2+x_imag.^2)+gamma4*(x_real.^4+2*x_real.^2.*x_imag.^2+x_imag.^4));
end

function integrand = integrand_moment2(x_real, x_imag, gamma0, gamma2, gamma4)
    integrand = (x_real.^2+x_imag.^2).*exp(gamma0 + gamma2*(x_real.^2+x_imag.^2)+gamma4*(x_real.^4+2*x_real.^2.*x_imag.^2+x_imag.^4));
end

function integrand = integrand_moment4(x_real, x_imag, gamma0, gamma2, gamma4)
    integrand = (x_real.^4+2*x_real.^2.*x_imag.^2+x_imag.^4).*exp(gamma0 + gamma2*(x_real.^2+x_imag.^2)+gamma4*(x_real.^4+2*x_real.^2.*x_imag.^2+x_imag.^4));
end

% Function to be solved
function F = system_of_equations(gammas, C_0, C_1, C_2)
    F(1) = integral2(@(x_real, x_imag) integrand_moment0(x_real, x_imag, gammas(1),gammas(2),gammas(3)),-Inf,Inf,-Inf,Inf, AbsTol=1e-10) - C_0;
    F(2) = integral2(@(x_real, x_imag) integrand_moment2(x_real, x_imag, gammas(1),gammas(2),gammas(3)),-Inf,Inf,-Inf,Inf, AbsTol=1e-10) - C_1;
    F(3) = integral2(@(x_real, x_imag) integrand_moment4(x_real, x_imag, gammas(1),gammas(2),gammas(3)),-Inf,Inf,-Inf,Inf, AbsTol=1e-10) - C_2;
end