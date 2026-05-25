clear;
clc;
close all;
N = 200;
n = 2;
m_list = [5, 10, 25, 100];
rho_list = [0, 0.5, 0.7, 0.9];
snr_dB_list = [0, 20];
dw_grid = linspace(0.02*pi, pi, 400);
for isnr = 1:length(snr_dB_list)
    snr_dB = snr_dB_list(isnr);
    snr = 10^(snr_dB / 10);
    sigma = 1 / snr;
    figure;
    for im = 1:length(m_list)
        m = m_list(im);
        subplot(2, 2, im);
        hold on;
        grid on;
        for irho = 1:length(rho_list)
            rho = rho_list(irho);
            P = [1, rho; rho, 1];
            eff = zeros(size(dw_grid));
            for idw = 1:length(dw_grid)
                dw = dw_grid(idw);
                w1 = -dw / 2;
                w2 = dw / 2;
                A = steering_matrix(m, [w1, w2]);
                D = steering_derivative_matrix(m, [w1, w2]);
                Pi_perp = eye(m) - A / (A' * A) * A';
                B = D' * Pi_perp * D;
                J = real(B .* transpose(P));
                CRB = sigma / (2 * N) * (J \ eye(n));
                Pinv = P \ eye(n);
                Ginv = (A' * A) \ eye(n);
                H = Pinv + sigma * Pinv * Ginv * Pinv;
                h1 = real(D(:, 1)' * Pi_perp * D(:, 1));
                var_MU = sigma / (2 * N) * real(H(1, 1)) / h1;
                eff(idw) = var_MU / real(CRB(1, 1));
            end
            plot(dw_grid / pi, eff, 'LineWidth', 1.5);
        end
        xlabel('\Delta\omega / \pi');
        ylabel('efficiency ratio');
        title(['m = ', num2str(m), ', SNR = ', num2str(snr_dB), ' dB']);
        legend('\rho=0', '\rho=0.5', '\rho=0.7', '\rho=0.9', 'Location', 'best');
        ylim([1, 10]);
    end
end
function A = steering_matrix(m, w)
    k = (0:m-1).';
    n = length(w);
    A = zeros(m, n);
    for i = 1:n
        A(:, i) = exp(1i * k * w(i));
    end
end
function D = steering_derivative_matrix(m, w)
    k = (0:m-1).';
    n = length(w);
    D = zeros(m, n);
    for i = 1:n
        D(:, i) = 1i * k .* exp(1i * k * w(i));
    end
end