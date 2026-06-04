clc;
clear all;
close all;

data_FER;

linewidth = 0.5;
markersize = 6;

h = figure;
fig(h, 'units','inches','width', 7, 'font','Times New Roman','fontsize', 14);

t = tiledlayout(2,2,'TileSpacing','none');

%%
nexttile;

semilogy(CAPolar64_snr_range, CAPolar64_SCL8, Marker="+", Color=[1 0 0], LineWidth=linewidth, MarkerSize=markersize);
hold on;
semilogy(CAPolar64_snr_range, CAPolar64_SCL32, Marker="o", Color=[0 1 0], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar64_snr_range, CAPolar64_SCL256, Marker="square", Color=[1 0 1], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar64_snr_range, CAPolar64_LC_OSD_DAI_delta8(15,:), Marker="^", Color=[0 0 0], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar64_snr_range, CAPolar64_LC_OSD_DAI_delta8(15,:), Marker="*", Color=[0 0 1], LineWidth=linewidth, MarkerSize=markersize);
% semilogy(CAPolar64_snr_range, CAPolar64_C_OSD_ML_delta8, Marker="none", LineStyle="--", Color=[0.5 0.5 0.5], LineWidth=linewidth, MarkerSize=markersize);

title('\large \rmfamily $\mathscr{C}_{\textrm{CA-polar}}[128,64]$', Interpreter='latex');

leg = legend([
"SCL$(8)$"
"SCL$(32)$"
"SCL$(256)$"
"LC-OSD$(8, 2^{14})$, SLVA"
"LC-OSD$(8, 2^{14})$, tFPT"
% "MLD lower bound"
]);

leg.Location = 'southwest';

grid on;
% xlabel('$E_{\textrm{b}}/N_0~(\textrm{dB})$');
ylabel('FER $\varepsilon$');
xticklabels({}) 
% yticklabels({}) 

h.CurrentAxes.XLabel.Interpreter = 'latex';
h.CurrentAxes.YLabel.Interpreter = 'latex';

axis([min(CAPolar64_snr_range) max(CAPolar64_snr_range) 1e-5 1]);

%%
nexttile

semilogy(CAPolar192_snr_range, CAPolar192_SCL8, Marker="+", Color=[1 0 0], LineWidth=linewidth, MarkerSize=markersize);
hold on;
semilogy(CAPolar192_snr_range, CAPolar192_SCL32, Marker="o", Color=[0 1 0], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar192_snr_range, CAPolar192_SCL256, Marker="square", Color=[1 0 1], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar192_snr_range, CAPolar192_LC_OSD_DAI_delta8(15,:), Marker="^", Color=[0 0 0], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar192_snr_range, CAPolar192_LC_OSD_DAI_delta8(15,:), Marker="*", Color=[0 0 1], LineWidth=linewidth, MarkerSize=markersize);
% semilogy(CAPolar192_snr_range, CAPolar192_LC_OSD_ML_delta8, Marker="none", LineStyle="--", Color=[0.5 0.5 0.5], LineWidth=linewidth, MarkerSize=markersize);


title('\large \rmfamily $\mathscr{C}_{\textrm{CA-polar}}[256,192]$', Interpreter='latex');
grid on;
% xlabel('$E_{\textrm{b}}/N_0~(\textrm{dB})$');
% ylabel('FER $\varepsilon$');
xticklabels({}) 
yticklabels({}) 

h.CurrentAxes.XLabel.Interpreter = 'latex';
h.CurrentAxes.YLabel.Interpreter = 'latex';

axis([min(CAPolar192_snr_range) max(CAPolar192_snr_range) 1e-5 1]);


%%
nexttile

semilogy(CAPolar64_snr_range, CAPolar64_SCL8_thr, Marker="+", Color=[1 0 0], LineWidth=linewidth, MarkerSize=markersize);
hold on;
semilogy(CAPolar64_snr_range, CAPolar64_SCL32_thr, Marker="o", Color=[0 1 0], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar64_snr_range, CAPolar64_SCL256_thr, Marker="square", Color=[1 0 1], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar64_snr_range, CAPolar64_LC_OSD_DAI_SLVA_delta8_thr, Marker="^", Color=[0 0 0], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar64_snr_range, CAPolar64_LC_OSD_DAI_tFPT_delta8_thr, Marker="*", Color=[0 0 1], LineWidth=linewidth, MarkerSize=markersize);


% leg = legend([
% "SCL$(8)$"
% "SCL$(32)$"
% "SCL$(256)$"
% "LC-OSD$(8, 2^{14})$ with tFPT"
% "LC-OSD$(8, 2^{14})$ with SLVA"
% % "MLD lower bound"
% ]);
% leg.Location = 'southeast';

grid on;
xlabel('$E_{\textrm{b}}/N_0~(\textrm{dB})$');
ylabel('Simulated throughput $(\textrm{bps})$');

h.CurrentAxes.XLabel.Interpreter = 'latex';
h.CurrentAxes.YLabel.Interpreter = 'latex';

axis([min(CAPolar64_snr_range) max(CAPolar64_snr_range) 3e3 2e6]);

%%
nexttile

semilogy(CAPolar192_snr_range, CAPolar192_SCL8_thr, Marker="+", Color=[1 0 0], LineWidth=linewidth, MarkerSize=markersize);
hold on;
semilogy(CAPolar192_snr_range, CAPolar192_SCL32_thr, Marker="o", Color=[0 1 0], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar192_snr_range, CAPolar192_SCL256_thr, Marker="square", Color=[1 0 1], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar192_snr_range, CAPolar192_LC_OSD_DAI_SLVA_delta8_thr, Marker="^", Color=[0 0 0], LineWidth=linewidth, MarkerSize=markersize);
semilogy(CAPolar192_snr_range, CAPolar192_LC_OSD_DAI_tFPT_delta8_thr, Marker="*", Color=[0 0 1], LineWidth=linewidth, MarkerSize=markersize);


grid on;
xlabel('$E_{\textrm{b}}/N_0~(\textrm{dB})$');
% ylabel('Simulated throughput $(\textrm{bps})$');
% xticklabels({}) 
yticklabels({}) 

h.CurrentAxes.XLabel.Interpreter = 'latex';
h.CurrentAxes.YLabel.Interpreter = 'latex';

axis([min(CAPolar192_snr_range) max(CAPolar192_snr_range) 3e3 2e6]);

%%

addpath("matlab2tikz-src\");
matlab2tikz('CAPolar-four-in-one.tex', 'height', '8.97cm', 'extraTikzpictureOptions', 'thick,scale=0.7, every node/.style={scale=0.7}', 'parseStrings', false);

%%
% Define the file name
filename = 'CAPolar-four-in-one.tex';

% Read the contents of the file
fileContents = fileread(filename);

% Perform the substitution
fileContents = replace(fileContents, 'width=', 'width=5.6cm,%');
fileContents = replace(fileContents, 'height=', 'height=4.3cm,%');
fileContents = replace(fileContents, 'title style=', '% title style=');
fileContents = replace(fileContents, 'ytick=', '% ytick=');
fileContents = replace(fileContents, 'yticklabels=', 'yticklabels={},%');

% Write the modified contents back to the file
fid = fopen(filename, 'w');
fwrite(fid, fileContents);
fclose(fid);