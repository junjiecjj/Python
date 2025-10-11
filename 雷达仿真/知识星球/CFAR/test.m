clear; close all; clc;

alpha = linspace(0, 20, 100);
N = 20;
pfa = zeros(1, length(alpha));
for ii = 1:length(alpha)
    pfa(ii) = compute_pfa(alpha(ii), N);
end

plot(alpha, pfa);
ylim([0, 0.001]);

function pfa = compute_pfa(alpha, N)
    % SOCA CFAR
    part1 = (2 + alpha / (N / 2)) ^ (-N / 2);
    part2 = 0;
    for k = 0:(N/2 - 1)
        part2 = part2 + nchoosek(N/2 - 1 + k, k) * (2 + alpha / (N / 2)) ^ (-k);
    end
    pfa = 2 * part1 * part2;

    % GOCA CFAR
    % part1 = (1 + alpha / (N / 2)) ^ (-N / 2);
    % part2 = (2 + alpha / (N / 2)) ^ (-N / 2);
    % part3 = 0;
    % for k = 0:(N/2 - 1)
    %     part3 = part3 + nchoosek(N/2 - 1 + k, k) * (2 + alpha / (N/2)) ^ (-k);
    % end
    % pfa = part1 - part2 * part3;
    % pfa = 2 * pfa;
end