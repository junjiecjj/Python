%% Detection Probability of a Target
disp('-----------------------------');
newTgtAz = -45;
disp(['Detection Probability of Target at ' num2str(newTgtAz) ' degree:']);
[~, idx] = min(abs(ang - newTgtAz));
theta_idx = 50;
disp(['     using Perfect Waveform : ' num2str(DetectProb(A_steer_vec(:, idx), Xs,     N0, 1e-5))]);
disp(['     using Nominal Waveform : ' num2str(DetectProb(A_steer_vec(:, idx), XBar,   N0, 1e-5))]);
disp(['     using Robust  Waveform : ' num2str(DetectProb(A_steer_vec(:, idx), Xr{theta_idx}, N0, 1e-5))]);
disp('-----------------------------');

%% Sum-Rate: Nominal Design
figure;
boxplot_bear(RealSumRateNominal, '', 0.5, [0.5804, 0, 0], 1, '-')    % my own boxplot function with a slight modification; Yes, I am Bear
xticks('');
ylabel('AASR (bps/Hz/user)');
set(gca, 'fontsize', 14);
hold on
yline(NominalSumRate, '-', 'Color', [0.1961, 0.8039, 0.1961], 'linewidth', 2);
legend({'Nominal Characterization'}, 'Interpreter', 'latex');
set(gca, 'YLim', [0.5 0.55])

%% Sum-Rate: Robust Design : Plot against all theta
figure;
theta_idx = 1:3:NumTheta;
theta = Theta(theta_idx);
Group = cell(length(theta), 1);
for i = 1:length(theta)
    Group{i} = num2str(theta(i), '%.2f');
end
boxplot_bear(RealSumRateRobust(:,theta_idx), Group, 0.5, [0.5804, 0, 0], 1, '-')
ylabel('AASR (bps/Hz/user)');
xlabel('$\theta$', 'interpreter', 'latex');
set(gca, 'fontsize', 14, 'xticklabelrotation', -45);
hold on
RobustSumRate_RepMat = repmat(RobustSumRate, NumTheta, 1);
boxplot_bear(RobustSumRate_RepMat(:, theta_idx), Group, 1.0, 'r', 2, '-')
% set(gca, 'YLim', [0.505 0.54])

%% Sum-Rate: Robust Design : Plot with a specified theta
figure;
radius = 0.125;
[~, theta_idx] = min(abs(Theta - radius));
theta = Theta(theta_idx);
boxplot_bear(RealSumRateRobust(:,theta_idx), '', 0.5, [0.5804, 0, 0], 1, '-')
xticks('');
ylabel('AASR (bps/Hz/user)');
set(gca, 'fontsize', 14);
hold on
yline(RobustSumRate(theta_idx), 'r', 'linewidth', 2);
if theta_idx == 1
    legend({'Nominal ($\theta$ = 0)'}, 'Interpreter', 'latex');
else
    legend({['Robust ($\theta$ = ' num2str(Theta(theta_idx), '%.3f') ')']}, 'Interpreter', 'latex');
end
% set(gca, 'YLim', [0.5 0.55])

%% Beampattern
figure;
radius = 0.127;
[~, theta_idx] = min(abs(Theta - radius));
plot(ang, pow2db(Bs/max(Bs)), 'k', 'LineWidth', 2);
hold on;
BBar = GetBeamPattern(XBar*XBar'/L, A_steer_vec);
plot(ang, pow2db(BBar/max(BBar)), 'b--', 'LineWidth', 2);
Br = GetBeamPattern(Xr{theta_idx}*Xr{theta_idx}'/L, A_steer_vec);
plot(ang, pow2db(Br/max(Br)), 'm-.', 'LineWidth', 2);
set(gca, 'fontsize', 16);
grid on;
xlabel('Azimuth (deg)');
ylabel('Beampattern (dB)');
legend({'Perfect', 'Nominal ($\theta$=0)', ['Robust ($\theta$=' num2str(Theta(theta_idx), '%.3f') ')']}, 'interpreter', 'latex');
ylim([-25 1]);

%%