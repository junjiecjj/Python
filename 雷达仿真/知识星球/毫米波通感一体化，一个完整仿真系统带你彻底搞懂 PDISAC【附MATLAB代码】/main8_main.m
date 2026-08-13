% MAIN8B_MAIN — PDISAC end-to-end driver (variant of main8_main).
% -------------------------------------------------------------------------
% Identical to main8_main EXCEPT for STEP 6a, which:
%   1. Reads exported_statistics/statistics_communication_b.csv (the
%      capacity-augmented CSV produced by main7_comm_analysis).
%   2. Restyles the BER figures (fig_ber_nsym, fig_ber_snr_nsym, fig_ber_snr):
%          Theoretical  ->  SOLID line   (was dashed)
%          Numerical    ->  DASHED line + markers  (markers kept, dashed line
%                           added to link them; was markers only)
%   3. REMOVES the net-throughput figure (fig_throughput_snr_nsym) and
%      replaces it with an ensemble-average CAPACITY figure that shows BOTH
%      units on a dual y-axis (left = bits/sequence, right = bits/s), using
%      the notation of the "Ensemble-Average Capacity Analysis" subsection
%      (5_7_results_comm.tex):
%          Cbar_seq = (1/N_prbs) sum_i sum_k log2(1+gamma_{k,i}) [bits/seq]  (eq_cap_seq_avg)
%          Cbar_net = Cbar_seq / T_prbs                          [bits/s]    (eq_cap_net)
%      The x-axis is N_bit^prbs (NOT SNR), matching the style of fig_ber_nsym.
%
% All other steps (pipeline STEP 1-5, channel-distribution STEP 6b) are the
% same as main8_main. Set NO_RESULT_PLOTS=true to skip STEP 6, or the
% NO_*_PLOT flags to run the pipeline headless.

% =====================================================================
%  STEP 0 — Setup: paths, GLOBAL figure style (set ONCE), output folder
% =====================================================================
clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));   % resolve all relocated scripts/functions

% Global LaTeX + Times New Roman rendering for EVERY figure. Set once here;
% downstream steps inherit these and must NOT repeat set(groot,...).
set(groot, 'defaultTextInterpreter',           'latex');
set(groot, 'defaultAxesTickLabelInterpreter',  'latex');
set(groot, 'defaultLegendInterpreter',         'latex');
set(groot, 'defaultColorbarTickLabelInterpreter','latex');
set(groot, 'defaultAxesFontName',              'Times New Roman');
set(groot, 'defaultTextFontName',              'Times New Roman');
set(groot, 'defaultAxesFontSize',              14);
set(groot, 'defaultLineLineWidth',             1.6);

fig_dir = fullfile(script_dir, 'fig_exported');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end

% =====================================================================
%  STEP 1 — System & scene configuration + geometry            (Sec. II)
% =====================================================================
% main2_topology calls main1_matlab_config internally (carrier, waveform,
% ROI, targets, scatterers, UE) and draws the scene plan view.
TOPO_NO_PLOT = false;
main2_topology;

% =====================================================================
%  STEP 2 — Transmit waveform & propagation channels           (Sec. II)
% =====================================================================
% ISAC waveform + sensing/comm channels -> received Y_sen (radar) and Y_com (UE).
main3_signal_channel_model;

% =====================================================================
%  STEP 3 — Radar sensing: MF -> RD map -> CA-CFAR -> interp   (Sec. III)
% =====================================================================
main4_sensing_process;

% =====================================================================
%  STEP 4 — Communication: delay/channel est. -> BPSK detect    (Sec. V)
% =====================================================================
main6_comm_process;

% =====================================================================
%  STEP 5 — Export the per-stage pipeline figures
% =====================================================================
figs = flipud(findall(0,'Type','figure'));
for k = 1:numel(figs)
    nm = get(figs(k),'Name');
    if isempty(nm), nm = sprintf('figure_%02d', k); end
    nm = matlab.lang.makeValidName(nm);
    savefig(figs(k),       fullfile(fig_dir, [nm '.fig']));
    exportgraphics(figs(k), fullfile(fig_dir, [nm '.png']), 'Resolution',300);
end
fprintf('Saved %d pipeline figure(s) to %s\n', numel(figs), fig_dir);

% =====================================================================
%  STEP 6 — Aggregate result figures (from exported statistics)
%  Rendered only if the statistics files exist (produced by
%  main7_comm_analysis / channel_com). Skip with NO_RESULT_PLOTS=true.
% =====================================================================
if ~exist('NO_RESULT_PLOTS','var') || ~NO_RESULT_PLOTS

% Okabe-Ito colour-blind-safe palette (shared by STEP 6a & 6b)
col.black  = [0 0 0];       col.orange = [230 159   0]/255;
col.green  = [  0 158 115]/255; col.yellow = [240 228  66]/255;
col.blue   = [  0 114 178]/255; col.vermil = [213  94   0]/255;
col.purple = [204 121 167]/255; col.gray   = [0.5 0.5 0.5];

% Line-style convention (this file): theory = solid, numerical = dashed+marker
LS_THEO = '-';   LW_THEO = 1.8;
LS_NUM  = '--';  LW_NUM  = 1.3;

% ---------------------------------------------------------------------
%  STEP 6a — Communication BER & capacity                      (Sec. V)
%  From exported_statistics/statistics_communication_b.csv:
%    Fig 1  BER vs. N_bit^prbs        (avg over UE, MC, SNR)
%    Fig 2  BER vs. SNR per N_bit^prbs (ideal + theory)
%    Fig 3  BER vs. SNR               (avg over UE, MC, N_bit^prbs)
%    Fig 4  Ensemble-average capacity vs. N_bit^prbs (bits/seq + bits/s)
%  Style: theory = solid line; numerical = dashed line + markers.
% ---------------------------------------------------------------------
comm_csv = fullfile(script_dir, 'exported_statistics', 'statistics_communication.csv');
if isfile(comm_csv)
    T = rmmissing(readtable(comm_csv));

    % Numerical BER variants: colour, marker, label
    numCol    = {col.blue, col.orange, col.green, col.vermil};
    numMk     = {'o','s','^','d'};
    numLbl    = { ...
        'Num.: $\hat{\tau}_{\mathrm{ue}}, \hat{h}_{\mathrm{com},i}$', ...
        'Num.: $\hat{\tau}_{\mathrm{ue}}, h_{\mathrm{com},i}$', ...
        'Num.: $\tau_{\mathrm{ue}}, \hat{h}_{\mathrm{com},i}$', ...
        'Num.: $\tau_{\mathrm{ue}}, h_{\mathrm{com},i}$'};
    numVarBER = {'Numerical_BER_Est_Tau_Est_h', 'Numerical_BER_Est_Tau_Perf_h', ...
                 'Numerical_BER_Perf_Tau_Est_h', 'Numerical_BER_Perf_Tau_Perf_h'};

    N_sym_list = unique(T.N_sym_per_block);
    N_sym_col  = {col.blue, col.orange, col.green, col.vermil, col.purple, col.yellow};

    % ---- Fig 1: BER vs. N_bit^prbs (avg over UE, MC, SNR) ------------
    G1 = groupsummary(T, 'N_sym_per_block', 'mean', ...
        [numVarBER, {'Theoretical_BER'}]);
    G1 = sortrows(G1, 'N_sym_per_block');

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    hold on; box on; grid on;
    for k = 1:4
        plot(G1.N_sym_per_block, G1.(['mean_' numVarBER{k}]), ...
            'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', numCol{k}, ...
            'Marker', numMk{k}, 'MarkerFaceColor', numCol{k}, 'MarkerSize', 7, ...
            'DisplayName', numLbl{k});
    end
    plot(G1.N_sym_per_block, G1.mean_Theoretical_BER, ...
        LS_THEO, 'Color', col.gray, 'LineWidth', LW_THEO, ...
        'DisplayName', 'Theo.: Analytical form');
    set(gca, 'XScale', 'log');
    xticks(N_sym_list); xticklabels(string(N_sym_list));
    xlabel('$N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    ylabel('Average Bit Error Rate (BER)');
    title('BER vs. $N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    legend('Location','southeast','Box','on','FontSize',11);
    xlim([min(N_sym_list) max(N_sym_list)]);
    savefig(gcf, fullfile(fig_dir,'fig_ber_nsym.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_ber_nsym.png'), 'Resolution',300);

    % ---- Fig 2: BER vs. SNR per N_bit^prbs (ideal + theory) ----------
    G2 = groupsummary(T, {'N_sym_per_block','SNR_dB'}, 'mean', ...
        {'Numerical_BER_Perf_Tau_Perf_h', 'Theoretical_BER'});

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    hold on; box on; grid on;
    for n = 1:numel(N_sym_list)
        Ns = N_sym_list(n);
        Gs = sortrows(G2(G2.N_sym_per_block == Ns, :), 'SNR_dB');
        c  = N_sym_col{mod(n-1, numel(N_sym_col)) + 1};
        % Numerical: dashed line + markers
        plot(Gs.SNR_dB, Gs.mean_Numerical_BER_Perf_Tau_Perf_h, ...
            'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', c, ...
            'Marker', 'o', 'MarkerFaceColor', c, 'MarkerSize', 6, ...
            'DisplayName', sprintf('$N_{\\mathrm{bit}}^{\\mathrm{prbs}} = %d$', Ns));
        % Theoretical: solid line
        plot(Gs.SNR_dB, Gs.mean_Theoretical_BER, ...
            LS_THEO, 'Color', c, 'LineWidth', LW_THEO, 'HandleVisibility', 'off');
    end
    % Dummy handles for the style legend (Num = dashed+marker, Theo = solid)
    plot(nan, nan, 'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', col.black, ...
        'Marker', 'o', 'MarkerFaceColor', col.black, ...
        'DisplayName', 'Num.: $\tau_{\mathrm{ue}}, h_{\mathrm{com},i}$');
    plot(nan, nan, LS_THEO, 'Color', col.black, 'LineWidth', LW_THEO, ...
        'DisplayName', 'Theo.: Analytical form');
    set(gca, 'YScale', 'log');
    xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
    ylabel('Average Bit Error Rate (BER)');
    title('BER vs. SNR for Different $N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    legend('Location','southwest','Box','on','FontSize',10,'NumColumns',2);
    savefig(gcf, fullfile(fig_dir,'fig_ber_snr_nsym.figsemilogy'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_ber_snr_nsym.png'), 'Resolution',300);

    % ---- Fig 3: BER vs. SNR (avg over UE, MC, N_bit^prbs) ------------
    G3 = groupsummary(T, 'SNR_dB', 'mean', ...
        [numVarBER, {'Theoretical_BER'}]);
    G3 = sortrows(G3, 'SNR_dB');

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    hold on; box on; grid on;
    for k = 1:4
        plot(G3.SNR_dB, G3.(['mean_' numVarBER{k}]), ...
            'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', numCol{k}, ...
            'Marker', numMk{k}, 'MarkerFaceColor', numCol{k}, 'MarkerSize', 7, ...
            'DisplayName', numLbl{k});
    end
    plot(G3.SNR_dB, G3.mean_Theoretical_BER, ...
        LS_THEO, 'Color', col.gray, 'LineWidth', LW_THEO, ...
        'DisplayName', 'Theo.: Analytical form');
    set(gca, 'YScale', 'log');
    xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
    ylabel('Average Bit Error Rate (BER)');
    title('BER vs. SNR');
    legend('Location','southwest','Box','on','FontSize',11);
    savefig(gcf, fullfile(fig_dir,'fig_ber_snr.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_ber_snr.png'), 'Resolution',300);

    % ---- Fig 4 (REPLACES fig_throughput_snr_nsym): --------------------
    %  Ensemble-average capacity vs. N_bit^prbs, both units on a dual
    %  y-axis. x-axis and style match Fig 1 (fig_ber_nsym): log x with
    %  ticks at N_sym_list, numerical = dashed+marker, theory = solid.
    %  Left  axis: Cbar_seq [bits/sequence]  (eq_cap_seq_avg)
    %  Right axis: Cbar_net [bits/s]          (eq_cap_net)
    %  Averaged over UE, MC, and SNR (grouped by N_bit^prbs), like Fig 1.
    capVars = {'Cap_Seq_Numerical','Cap_Seq_Theoretical', ...
               'Cap_Net_Numerical','Cap_Net_Theoretical'};
    GC = groupsummary(T, 'N_sym_per_block', 'mean', capVars);
    GC = sortrows(GC, 'N_sym_per_block');

    % Recover T_prbs from the exact proportionality Cbar_net = Cbar_seq/T_prbs.
    ratio  = T.Cap_Seq_Theoretical ./ T.Cap_Net_Theoretical;
    ratio  = ratio(isfinite(ratio) & ratio > 0);
    T_prbs = median(ratio);

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');

    % --- left axis: bits/sequence ---
    yyaxis(ax,'left');
    plot(ax, GC.N_sym_per_block, GC.mean_Cap_Seq_Numerical, ...
        'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', col.blue, ...
        'Marker', 'o', 'MarkerFaceColor', col.blue, 'MarkerSize', 7, ...
        'DisplayName', 'Num.: Monte-Carlo');
    plot(ax, GC.N_sym_per_block, GC.mean_Cap_Seq_Theoretical, ...
        LS_THEO, 'Color', col.gray, 'LineWidth', LW_THEO, ...
        'DisplayName', 'Theo.: Analytical form');
    ylabel(ax, '$\bar{\mathcal{C}}_{\mathrm{seq}}$ (bits/sequence)');
    ax.YAxis(1).Color = col.black;

    % --- right axis: bits/second (left axis rescaled by 1/T_prbs) ---
    yyaxis(ax,'right');
    ylabel(ax, '$\bar{\mathcal{C}}_{\mathrm{net}}$ (bits/s)');
    ax.YAxis(2).Color = col.black;

    yyaxis(ax,'left');  yl = ylim(ax);
    yyaxis(ax,'right'); ylim(ax, yl / T_prbs);
    yyaxis(ax,'left');   % keep left active for the curves/legend

    set(ax, 'XScale', 'log');
    xticks(ax, N_sym_list); xticklabels(ax, string(N_sym_list));
    xlim(ax, [min(N_sym_list) max(N_sym_list)]);
    xlabel(ax, '$N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    title(ax, 'Ensemble-Average Capacity vs. $N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    legend(ax, 'Location','northwest','Box','on','FontSize',11);
    savefig(gcf, fullfile(fig_dir,'fig_cap_nsym.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_cap_nsym.png'), 'Resolution',300);

    fprintf('STEP 6a: 3 BER figures + 1 capacity figure saved to %s (T_prbs = %.4g s)\n', ...
        fig_dir, T_prbs);
else
    warning('main8b_main:skipCommPerf', ...
        'STEP 6a skipped: %s not found (run main7b_comm_analysis first).', comm_csv);
end

% ---------------------------------------------------------------------
%  STEP 6c — Sensing RMSE parameter sweeps (from main5b CSVs)   (Sec. IV)
%  Reads the sweep statistics written by main5b_sensing_analysis:
%    Fig 5  Range RMSE vs. SNR for each N_chip (statistics_sensing_sweep_nchip.csv)
%    Fig 6  Velocity RMSE vs. SNR for each N_prbs (statistics_sensing_sweep_nprbs.csv)
%  Each swept value is a numerical curve (dashed line + markers) with its
%  95% bootstrap CI as a shaded band, drawn in the same main8b style.
%  Self-contained colours/markers so this block runs even if STEP 6a was
%  skipped.
% ---------------------------------------------------------------------
sweepCol = {col.blue, col.orange, col.green, col.vermil, col.purple, col.yellow};
sweepMk  = {'o','s','^','d','v','>'};

sweep_nchip_csv = fullfile(script_dir, 'exported_statistics', 'statistics_sensing_sweep_nchip.csv');
sweep_nprbs_csv = fullfile(script_dir, 'exported_statistics', 'statistics_sensing_sweep_nprbs.csv');

% ---- Fig 5: Range RMSE vs. SNR for each N_chip (N_prbs fixed) ----------
if isfile(sweep_nchip_csv)
    A = rmmissing(readtable(sweep_nchip_csv));
    nchip_list = unique(A.N_chip);

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    hold on; box on; grid on;
    for n = 1:numel(nchip_list)
        Nc = nchip_list(n);
        As = sortrows(A(A.N_chip == Nc, :), 'SNR_dB');
        c  = sweepCol{mod(n-1, numel(sweepCol)) + 1};
        xr = As.SNR_dB;
        lo = As.RMSE_Range_m_CI_lo; hi = As.RMSE_Range_m_CI_hi;
        meanRMSE = (lo + hi) / 2;   % mean of the 95% CI bounds (not the CI mid/point estimate)
        % 95% bootstrap CI band (no legend entry)
        vd = ~isnan(lo) & ~isnan(hi);
        fill([xr(vd); flipud(xr(vd))], [lo(vd); flipud(hi(vd))], c, ...
            'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        % Numerical mean curve: dashed line + markers
        plot(xr, meanRMSE, 'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', c, ...
            'Marker', sweepMk{mod(n-1, numel(sweepMk)) + 1}, ...
            'MarkerFaceColor', c, 'MarkerSize', 6, ...
            'DisplayName', sprintf('$N_{\\mathrm{chip}} = %d$', Nc));
    end
    set(gca, 'YScale', 'log');
    xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
    ylabel('Range RMSE (m)');
    title('Range RMSE vs. SNR for Different $N_{\mathrm{chip}}$');
    legend('Location','northeast','Box','on','FontSize',11);
    savefig(gcf, fullfile(fig_dir,'fig_rmse_range_vs_snr_nchip_sweep.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_rmse_range_vs_snr_nchip_sweep.png'), 'Resolution',300);
else
    warning('main8b_main:skipNchipSweep', ...
        'STEP 6c range sweep skipped: %s not found (run main5b_sensing_analysis first).', ...
        sweep_nchip_csv);
end

% ---- Fig 6: Velocity RMSE vs. SNR for each N_prbs (N_chip fixed) -------
if isfile(sweep_nprbs_csv)
    Bv = rmmissing(readtable(sweep_nprbs_csv));
    nprbs_list = unique(Bv.N_prbs);

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    hold on; box on; grid on;
    for n = 1:numel(nprbs_list)
        Np = nprbs_list(n);
        Bs = sortrows(Bv(Bv.N_prbs == Np, :), 'SNR_dB');
        c  = sweepCol{mod(n-1, numel(sweepCol)) + 1};
        xr = Bs.SNR_dB;
        lo = Bs.RMSE_Vel_mps_CI_lo; hi = Bs.RMSE_Vel_mps_CI_hi;
        meanRMSE = (lo + hi) / 2;   % mean of the 95% CI bounds (not the CI mid/point estimate)
        vd = ~isnan(lo) & ~isnan(hi);
        fill([xr(vd); flipud(xr(vd))], [lo(vd); flipud(hi(vd))], c, ...
            'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(xr, meanRMSE, 'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', c, ...
            'Marker', sweepMk{mod(n-1, numel(sweepMk)) + 1}, ...
            'MarkerFaceColor', c, 'MarkerSize', 6, ...
            'DisplayName', sprintf('$N_{\\mathrm{prbs}} = %d$', Np));
    end
    set(gca, 'YScale', 'log');
    xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
    ylabel('Velocity RMSE (m/s)');
    title('Velocity RMSE vs. SNR for Different $N_{\mathrm{prbs}}$');
    legend('Location','northeast','Box','on','FontSize',11);
    savefig(gcf, fullfile(fig_dir,'fig_rmse_vel_vs_snr_nprbs_sweep.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_rmse_vel_vs_snr_nprbs_sweep.png'), 'Resolution',300);
else
    warning('main8b_main:skipNprbsSweep', ...
        'STEP 6c velocity sweep skipped: %s not found (run main5b_sensing_analysis first).', ...
        sweep_nprbs_csv);
end

% ---------------------------------------------------------------------
%  STEP 6b — Channel LOS/NLOS distribution                     (Sec. II)
%  From exported_statistics/statistics_channel.mat (does NOT re-run the sim).
%  Each channel is z-scored by its own std, mapping both onto a common
%  "number of sigmas" axis: LOS -> arcsine on [-sqrt2, sqrt2], NLOS -> N(0,1).
% ---------------------------------------------------------------------
chan_mat = fullfile(script_dir, 'exported_statistics', 'statistics_channel.mat');
if isfile(chan_mat)
    S = load(chan_mat, 'los_channel_all', 'nlos_channel_all');

    % Pool real + imaginary parts (same distribution by symmetry)
    los_pooled  = [real(S.los_channel_all);  imag(S.los_channel_all)];
    nlos_pooled = [real(S.nlos_channel_all); imag(S.nlos_channel_all)];
    los_pooled  = los_pooled(isfinite(los_pooled));
    nlos_pooled = nlos_pooled(isfinite(nlos_pooled));

    % z-score each channel by its own mean/std
    los_z  = (los_pooled  - mean(los_pooled, 'omitnan'))  / std(los_pooled, 'omitnan');
    nlos_z = (nlos_pooled - mean(nlos_pooled,'omitnan')) / std(nlos_pooled,'omitnan');

    % Theoretical shapes in z-units: LOS arcsine (Var = A^2/2 = 1 -> A = sqrt2), NLOS N(0,1)
    x_grid  = linspace(-4.2, 4.2, 4000)';
    y_arc   = arcsine_pdf(x_grid, sqrt(2));
    y_gauss = gaussian_pdf(x_grid, 0, 1);

    col_los_fit  = [0 40 90]/255;    % dark blue (arcsine fit)
    col_nlos_fit = [90 30 0]/255;    % dark vermillion (gaussian fit)

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    histogram(ax, los_z,  'Normalization','pdf', 'NumBins',150, ...
        'FaceColor', col.blue,   'EdgeColor','none', 'FaceAlpha',0.55, ...
        'DisplayName','LOS (empirical)');
    histogram(ax, nlos_z, 'Normalization','pdf', 'NumBins',150, ...
        'FaceColor', col.vermil, 'EdgeColor','none', 'FaceAlpha',0.55, ...
        'DisplayName','NLOS (empirical)');
    plot(ax, x_grid, y_arc,   '--', 'Color', col_los_fit,  'LineWidth', 2.2, ...
        'DisplayName','LOS fit: arcsine');
    plot(ax, x_grid, y_gauss, '--', 'Color', col_nlos_fit, 'LineWidth', 2.2, ...
        'DisplayName','Aggregate NLOS fit: Gaussian');
    set(ax, 'YScale','log'); ylim(ax,[1e-3 1e1]); xlim(ax,[-4.2 4.2]);
    xlabel(ax, 'Amplitude');
    ylabel(ax, 'PDF (log scale)');
    title(ax, '$h_{\mathrm{com}}(t) = h^{\mathrm{los}}_{\rm com}(t) + h^{\mathrm{nlos}}_{\rm com}(t)$');
    legend(ax, 'Location','north','Box','on','FontSize',11,'NumColumns',2);
    savefig(gcf, fullfile(fig_dir,'fig_com_channel_dis.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_com_channel_dis.png'), 'Resolution',300);
    fprintf('STEP 6b: channel-distribution figure saved to %s\n', fig_dir);
else
    warning('main8b_main:skipChannelDist', ...
        'STEP 6b skipped: %s not found (run channel_com first).', chan_mat);
end

end   % NO_RESULT_PLOTS


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function y = arcsine_pdf(x, A)
    % PDF of x = A*sin(theta), theta ~ Uniform(0,2*pi): U-shaped ("arcsine")
    % distribution on [-A, A].
    y = zeros(size(x));
    inside = abs(x) < A;
    y(inside) = 1 ./ (pi * sqrt(A.^2 - x(inside).^2));
end

function y = gaussian_pdf(x, mu, sigma)
    y = 1 ./ (sigma .* sqrt(2*pi)) .* exp(-0.5 * ((x - mu) ./ sigma).^2);
end
