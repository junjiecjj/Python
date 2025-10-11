%% Detection Probability of a Target
disp('-----------------------------');
newTgtAz = -40;
disp(['Detection Probability of Target at ' num2str(newTgtAz) ' degree:']);
[~, idx] = min(abs(ang - newTgtAz));
rho_inx   = 1;
theta_idx = 50;
disp(['     using Perfect Waveform : ' num2str(DetectProb(A_steer_vec(:, idx), Xs,     N0, 1e-5))]);
disp(['     using Nominal Waveform : ' num2str(DetectProb(A_steer_vec(:, idx), XBar{rho_inx},   N0, 1e-5))]);
disp(['     using Robust  Waveform : ' num2str(DetectProb(A_steer_vec(:, idx), Xr{rho_inx, theta_idx}, N0, 1e-5))]);
disp('-----------------------------');

%% Sum-Rate: Nominal Design
figure;
PlotMode = 2;
switch PlotMode
    case 1
        rho    = 0.53;
        [~, rho_idx]   = min(abs(Rho - rho));

        RealSumRateNominal_Part = RealSumRateNominal(rho_idx, :);
        boxplot_bear(RealSumRateNominal_Part', '', 0.5, [0.5804, 0, 0], 1, '-')    % my own boxplot function with a slight modification; Yes, I am Bear
        xticks('');
        ylabel('AASR (bps/Hz/user)');
        if strcmp(DesignType, 'JOINT')
            title(['$\rho = $ ' num2str(Rho(rho_idx), '%.2f') ', ' PowerType], 'interpreter', 'latex');
        end
        set(gca, 'fontsize', 16);
        hold on
        yline(NominalSumRate(rho_idx), '-', 'color', [0.1961, 0.8039, 0.1961], 'linewidth', 2)

        legend({'Nominal Characterization'}, 'Interpreter', 'latex');

    case 2
        range   = 1:11;
        range   = 1:length(Rho);
        Group = cell(length(Rho(range)), 1);
        for i = range
            Group{i} = num2str(Rho(i), '%.2f');
        end
        
        boxplot_bear(RealSumRateNominal(range, :)', Group, 0.5, [0.5804, 0, 0], 1, '-')    % my own boxplot function with a slight modification; Yes, I am Bear
        xticks('');
        xlabel('$\rho$', 'interpreter', 'latex')
        ylabel('AASR (bps/Hz/user)');
        set(gca, 'fontsize', 16);
        hold on
        NominalSumRate_RepMat = repmat(NominalSumRate(range)', NumRho, 1);
        boxplot_bear(NominalSumRate_RepMat, Group, 1.0, [0.1961, 0.8039, 0.1961], 2, '-')
end
% set(gca, 'YLim', [1.85 2.5])

%% Sum-Rate: Robust Design : Plot against all theta upon specifying rho
figure;
rho    = 0.25;
[~, rho_idx]   = min(abs(Rho - rho));

theta_idx = 1:3:NumTheta;
theta = Theta(theta_idx);

Group = cell(length(theta), 1);
for i = 1:length(theta)
    Group{i} = num2str(theta(i), '%.2f');
end

boxplot_bear(RealSumRateRobust{rho_idx}(:, theta_idx), Group, 0.5, [0.5804, 0, 0], 1, '-')
ylabel('AASR (bps/Hz/user)');
xlabel('$\theta$', 'interpreter', 'latex');
if strcmp(DesignType, 'JOINT')
    title(['$\rho = $ ' num2str(Rho(rho_idx), '%.2f') ', ' PowerType], 'interpreter', 'latex');
end
set(gca, 'fontsize', 14, 'xticklabelrotation', -45);
hold on
RobustSumRate_RepMat = repmat(RobustSumRate(rho_idx, :), NumTheta, 1);
boxplot_bear(RobustSumRate_RepMat(:, theta_idx), Group, 1.0, 'r', 2, '-')
% set(gca, 'YLim', [0.505 0.54])

%% Sum-Rate: Robust Design : Plot against all rho upon specifying theta
figure;
radius    = 0.16;
[~, theta_idx]   = min(abs(Theta - radius));

rho_idx = 1:10:101;
rho = Rho(rho_idx);

RealSumRateRobust_Local = zeros(MC, length(rho));
Group = cell(length(rho), 1);
for i = 1:length(rho)
    Group{i} = num2str(rho(i), '%.2f');
    RealSumRateRobust_Local(:, i) = RealSumRateRobust{rho_idx(i)}(:, theta_idx);
end

boxplot_bear(RealSumRateRobust_Local, Group, 0.5, [0.5804, 0, 0], 1, '-')
ylabel('AASR (bps/Hz/user)');
xlabel('$\rho$', 'interpreter', 'latex');
if strcmp(DesignType, 'JOINT')
    title(['$\theta = $ ' num2str(Theta(theta_idx), '%.2f') ', ' PowerType], 'interpreter', 'latex');
end
set(gca, 'fontsize', 14, 'xticklabelrotation', -45);
hold on
RobustSumRate_RepMat = repmat(RobustSumRate(rho_idx, theta_idx)', length(rho), 1);
boxplot_bear(RobustSumRate_RepMat, Group, 1.0, 'r', 2, '-')
% set(gca, 'YLim', [0.505 0.54])

%% Sum-Rate: Robust Design : Plot with a specified theta upon specifying rho
figure;
radius = 0.16;
[~, theta_idx] = min(abs(Theta - radius));

rho    = 0.25;
[~, rho_idx]   = min(abs(Rho - rho));

theta = Theta(theta_idx);
boxplot_bear(RealSumRateRobust{rho_idx}(:,theta_idx), '', 0.5, [0.5804, 0, 0], 1, '-')
xticks('');
ylabel('AASR (bps/Hz/user)');
if strcmp(DesignType, 'JOINT')
    title(['$\rho = $ ' num2str(Rho(rho_idx), '%.2f') ', ' PowerType], 'interpreter', 'latex');
end
set(gca, 'fontsize', 14);
hold on
yline(RobustSumRate(rho_idx, theta_idx), 'r', 'linewidth', 2.5);
if theta_idx == 1
    legend({'Nominal ($\theta$ = 0)'}, 'Interpreter', 'latex');
else
    legend({['Robust ($\theta$ = ' num2str(theta, '%.2f') ')']}, 'Interpreter', 'latex');
end
% set(gca, 'YLim', [0.5 0.55])

%% Beampattern
figure;
radius = 0.16;
rho    = 0.25;
[~, theta_idx] = min(abs(Theta - radius));
[~, rho_idx]   = min(abs(Rho - rho));

plot(ang, pow2db(Bs/max(Bs)), 'k', 'LineWidth', 2);
hold on;
BBar = GetBeamPattern(XBar{rho_idx}*XBar{rho_idx}'/L, A_steer_vec);
plot(ang, pow2db(BBar/max(BBar)), 'b--', 'LineWidth', 2);
Br = GetBeamPattern(Xr{rho_idx, theta_idx}*Xr{rho_idx, theta_idx}'/L, A_steer_vec);
plot(ang, pow2db(Br/max(Br)), 'm-.', 'LineWidth', 2);
set(gca, 'fontsize', 16);
grid on;
xlabel('Azimuth (deg)');
ylabel('Beampattern (dB)');
legend({'Perfect', 'Nominal ($\theta$=0)', ['Robust ($\theta$=' num2str(Theta(theta_idx), '%.2f') ')']}, 'interpreter', 'latex');
ylim([-25 1]);
if strcmp(DesignType, 'JOINT')
    title(['$\rho = $ ' num2str(Rho(rho_idx), '%.2f') ', ' PowerType], 'interpreter', 'latex');
end

%%