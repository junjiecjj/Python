function plot_constellation(symHat, symRef, waveformName, ber)
%PLOT_CONSTELLATION 画均衡后星座图

figure('Name', [waveformName, ' Constellation'], 'Color', 'w');
scatter(real(symHat), imag(symHat), 18, 'filled');
hold on;
plot(real(symRef), imag(symRef), 'rp', 'MarkerSize', 10, 'LineWidth', 1.2);
grid on;
axis equal;
xlim([-2, 2]);
ylim([-2, 2]);
xlabel('In-Phase');
ylabel('Quadrature');
legend('Equalized symbols', 'Ideal QPSK points', 'Location', 'best');
title(sprintf('%s equalized constellation, BER = %.4e', waveformName, ber));
end
