function plot_effective_channel(Heff, waveformName)
%PLOT_EFFECTIVE_CHANNEL 画等效通信信道热力图

Hmag = abs(Heff);
HmagdB = 20 * log10(Hmag / (max(Hmag(:)) + eps) + eps);

figure('Name', [waveformName, ' Effective Channel'], 'Color', 'w');
% imagesc(HmagdB);
mesh(HmagdB);
axis xy;
colormap(turbo);
colorbar;
% caxis([-40, 0]);
xlabel('Transmit symbol index');
ylabel('Receive symbol index');
title(sprintf('%s equivalent communication channel |H_{eff}| (dB)', waveformName), ...
    'Interpreter', 'tex');
end
