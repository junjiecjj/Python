%{
    Online supplementary materials of the paper titled:
    Distributionally Robust Adaptive Beamforming
    By Shixiong Wang, Wei Dai, and Geoffrey Ye Li
    From Department of Electrical and Electronic Engineering, Imperial College London

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     8 Oct 2024, 13 March 2025
    @Home:     https://github.com/Spratm-Asleaf/Beamforming-UDL
%}


%% Plot Output SINR Results
colors = [
    0.9290, 0.6940, 0.1250     % Orange
    0.6000, 0     , 0.2000     % Dark Red
    0.5800, 0.0000, 0.8270     % Vivid Violet
];

figure;
hold on;

legend_text = {};

if isPlotAgainstSNR
    xAxisLabel = SNR;
else
    xAxisLabel = Snapshot;
end

plot(xAxisLabel, mean(SINR_IPN0, 1),       'k'                  ,     'LineWidth', 2);
legend_text = [legend_text, 'Optimal'];

plot(xAxisLabel, mean(SINR_IPN, 1),        'b'                  ,     'LineWidth', 2);
legend_text = [legend_text, 'IPN'];

plot(xAxisLabel, mean(SINR_IPN_DL, 1),     'g'                  ,     'LineWidth', 2);
legend_text = [legend_text, 'IPN-DL'];

plot(xAxisLabel, mean(SINR_IPN_UDL, 1),    'r'                  ,      'LineWidth', 2);
legend_text = [legend_text, 'IPN-UDL'];

plot(xAxisLabel, mean(SINR_IPN_MatEnt, 1), 'm'                  ,     'LineWidth', 2);
legend_text = [legend_text, 'IPN-MatEnt'];

% Totally amounts to the "Optimal Beamformer" (i.e., IPN0) before
% plot(xAxisLabel, mean(SINR_Capon0, 1),     'b--'              ,     'LineWidth', 2);
% legend_text = [legend_text, 'Capon0'];

plot(xAxisLabel, mean(SINR_Capon, 1),      'Color', colors(1, :),     'LineWidth', 2);
legend_text = [legend_text, 'Capon'];

plot(xAxisLabel, mean(SINR_Capon_DL, 1),   'Color', colors(2, :),     'LineWidth', 2);
legend_text = [legend_text, 'Capon-DL'];

legend(legend_text, 'Location', 'northwest');

axis([xAxisLabel(1) xAxisLabel(end) -10 40]);

box on;

set(gca, 'fontsize', 16);

ylabel('Output SINR (dB)');
if isPlotAgainstSNR
    xlabel('Input SNR (dB)');
else
    xlabel('Snapshot');
end

