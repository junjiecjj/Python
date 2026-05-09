function plot_caf_map(CAF, delayGrid, dopplerGrid, trueDelays, trueDopplers, waveformName)
%PLOT_CAF_MAP 画CAF，并标注真实目标

CAFmagdB = 20 * log10(abs(CAF) / (max(abs(CAF(:))) + eps) + eps);

figure('Name', [waveformName, ' CAF'], 'Color', 'w');
% imagesc(delayGrid, dopplerGrid, CAFmagdB);
mesh(delayGrid, dopplerGrid, CAFmagdB);
axis xy;
colormap(turbo);
colorbar;
% caxis([-35, 0]);
xlabel('Delay bin');
ylabel('Doppler bin');
title(sprintf('%s cross-ambiguity function (CAF)', waveformName));
hold on;

for p = 1:numel(trueDelays)
    plot(trueDelays(p), trueDopplers(p), 'wp', ...
        'MarkerSize', 13, 'MarkerFaceColor', 'k', 'LineWidth', 1.3);
    text(trueDelays(p) + 0.5, trueDopplers(p) + 0.3, sprintf('T%d', p), ...
        'Color', 'w', 'FontWeight', 'bold', 'FontSize', 11);
end

end
