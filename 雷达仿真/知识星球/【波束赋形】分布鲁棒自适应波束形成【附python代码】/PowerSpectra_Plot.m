%{
    Online supplementary materials of the paper titled:
    Distributionally Robust Adaptive Beamforming
    By Shixiong Wang, Wei Dai, and Geoffrey Ye Li
    From Department of Electrical and Electronic Engineering, Imperial College London

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     8 Oct 2024, 13 March 2025
    @Home:     https://github.com/Spratm-Asleaf/Beamforming-UDL
%}


% Plot
plotColor  = 'g';
refColor   = 'r';
ax         = [-35 -15 0 1];

avrgLineWidth    = 2.0;
mcArvgLineWidth  = 3.0;
mcLineWidth      = 1.0;

% close all;

%% Plot Avrg

figure('Position', [680   458   560   420]);
hold on;
legend_text = {};

% tempSpectra = GetAvrgSpectra(SpectraCapon0);
% plot(ThetaSweep*180/pi, tempSpectra, 'r', 'linewidth', avrgLineWidth);
% legend_text = [legend_text, 'Capon-0'];

tempSpectra = GetAvrgSpectra(SpectraCapon);
plot(ThetaSweep*180/pi, tempSpectra, 'b', 'linewidth', avrgLineWidth);
legend_text = [legend_text, 'Capon'];

tempSpectra = GetAvrgSpectra(SpectraCaponDL);
plot(ThetaSweep*180/pi, tempSpectra, 'g', 'linewidth', avrgLineWidth);
legend_text = [legend_text, 'Capon-DL'];

tempSpectra = GetAvrgSpectra(SpectraCaponUDL);
plot(ThetaSweep*180/pi, tempSpectra, 'r', 'linewidth', avrgLineWidth);
legend_text = [legend_text, 'Capon-UDL'];

% Perforance not stable from one trial to another; be cautious using this method
tempSpectra = GetAvrgSpectra(SpectraMaxEnt);
plot(ThetaSweep*180/pi, tempSpectra, 'k', 'linewidth', avrgLineWidth);
legend_text = [legend_text, 'MaxEnt'];
% 
% tempSpectra = GetAvrgSpectra(SpectraMusic);
% plot(ThetaSweep*180/pi, tempSpectra, 'c', 'linewidth', avrgLineWidth);
% legend_text = [legend_text, 'MUSIC'];

for i = 1:length(Theta)
    xline(Theta(i)*180/pi, 'k--', 'linewidth', 1);
end
legend(legend_text);

set(gca, 'fontsize', 16);
ylabel('Power Pattern');
xlabel('Angle (Degree)');

axis(ax);

box on;
