%% pulse_train_spectrum.m
% 相参 vs 非相参 脉冲串频谱对比（不分段；同时显示并保存）
clear; clc; close all; rng(1);
 
%% ================= 信号与频谱参数 =================
fs      = 200e6;          % 采样率 [Hz]
PRI     = 100e-6;         % 脉冲重复间隔 [s]
PRF     = 1/PRI;
tau     = 5e-6;           % 脉宽 [s]
Npulse  = 128;            % 脉冲数
fc      = 0;              % 基带演示（设非零可搬移到载频）
 
% 非相参扰动
use_random_phase = true;
use_PRI_jitter   = true;
use_amp_jitter   = true;
sigma_phi = pi;               % 相位抖动
sigma_PRI = 0.02*PRI;         % 2% PRI 抖动
sigma_amp = 0.05;             % 5% 幅度抖动
 
% 频谱
pad_factor   = 2;
zoom_bins    = 2;            % "±N×PRF 细节"中的 N
BAND_VIEW_HZ = 2e5;           % 周期图仅显示 ±BAND_VIEW_HZ
 
%% ================= 绘图/导出控制 =================
OUTDIR         = "out_show_and_save";
RES_DPI        = 220;         % 导出分辨率
Y_LIM_BAND     = [-120 5];    % 周期图 y 轴
Y_LIM_WELCH    = [-120 5];    % Welch y 轴
MAXPTS_MAIN    = 8e4;         % 周期图主曲线抽点上限（保极值）
MAXPTS_REF     = 3e4;         % 参考曲线抽点上限
RENDERER       = 'opengl';    % 屏显更流畅：'opengl'；导出矢量可用 'painters'
SAVE_PDF       = false;       % 同名 PDF（矢量），如需请置 true
SAVE_FIG       = false;       % 同名 FIG（MATLAB 可编辑），如需请置 true
 
if ~exist(OUTDIR,'dir'), mkdir(OUTDIR); end
 
%% ================= 生成信号 =================
T_total = Npulse*PRI;
Ns      = round(T_total*fs);
t       = (0:Ns-1).'/fs;
x_coh   = complex(zeros(Ns,1));
x_ncoh  = complex(zeros(Ns,1));
L_p     = max(1, round(tau*fs));
 
for n = 0:Npulse-1
    t0_nom = n*PRI;
    t0 = t0_nom + (use_PRI_jitter)*randn()*sigma_PRI;
    k0 = round(t0*fs)+1;  k1 = k0+L_p-1;
    if k1<1 || k0>Ns, continue; end
    k0=max(k0,1); k1=min(k1,Ns); seg=k0:k1;
 
    A = 1 + (use_amp_jitter)*sigma_amp*randn();
    phi_coh = 0;
    if use_random_phase
        if isfinite(sigma_phi) && sigma_phi>0
            phi_ncoh = max(-pi, min(pi, randn()*sigma_phi));
        else
            phi_ncoh = 2*pi*rand();
        end
    else
        phi_ncoh = 0;
    end
 
    tt = t(seg);
    x_coh(seg)  = x_coh(seg)  + A*exp(1j*(2*pi*fc*tt + phi_coh));
    x_ncoh(seg) = x_ncoh(seg) + A*exp(1j*(2*pi*fc*tt + phi_ncoh));
end
 
%% ================= 周期图 & 参考曲线 =================
Nfft = 2^nextpow2(Ns*pad_factor);
Xc   = fftshift(fft(x_coh,  Nfft));
Xn   = fftshift(fft(x_ncoh, Nfft));
f    = ((-Nfft/2):(Nfft/2-1)).'*(fs/Nfft);   % Hz
 
Pc_dB = 10*log10( (abs(Xc)/Ns).^2 + eps );
Pn_dB = 10*log10( (abs(Xn)/Ns).^2 + eps );
ref0  = max(Pc_dB);  Pc_dB = Pc_dB - ref0;  Pn_dB = Pn_dB - ref0;
 
% 单脉冲包络 + Dirichlet（视觉参考）
env_dB = 20*log10( abs(sinc(f*tau)) + eps ); env_dB = env_dB - max(env_dB);
T = PRI;
dirich = Npulse * (abs(sinc(Npulse*f*T)) ./ (abs(sinc(f*T))+eps));
dirich = dirich/max(dirich);
dir_dB = 20*log10(dirich + eps);
theo_coh_env_dB = env_dB + dir_dB;
 
%% ================= Welch PSD =================
win    = hann(4096);
nover  = floor(0.5*numel(win));
nfft_w = 8192;
[Pw_c, fw] = pwelch(x_coh,  win, nover, nfft_w, fs, 'centered','power');
[Pw_n, ~ ] = pwelch(x_ncoh, win, nover, nfft_w, fs, 'centered','power');
Pw_c_dB = 10*log10(Pw_c + eps);
Pw_n_dB = 10*log10(Pw_n + eps);
Pw_c_dB = Pw_c_dB - max(Pw_c_dB);
Pw_n_dB = Pw_n_dB - max(Pw_c_dB);
 
%% ================= 1) 时域（前三个 PRI） =================
fig = figure('Visible','on','Color','w','Position',[60 80 1100 320],'Renderer',RENDERER);
plot(t*1e3, real(x_coh), 'LineWidth',1); hold on;
plot(t*1e3, real(x_ncoh), 'LineWidth',1);
grid on; xlim([0 3*PRI*1e3]);
xlabel('Time [ms]'); ylabel('Amplitude');
title('时域（放大到前三个 PRI）'); legend('相参','非相参','Location','best');
save_figure(fig, fullfile(OUTDIR,'01_time_domain'), RES_DPI, SAVE_PDF, SAVE_FIG);
 
%% ================= 2) 周期图（仅 ±BAND_VIEW_HZ） ==========
idx_band = abs(f) <= BAND_VIEW_HZ;
f_band   = f(idx_band); Pc_band = Pc_dB(idx_band); Pn_band = Pn_dB(idx_band);
env_band = env_dB(idx_band);   ref_band = theo_coh_env_dB(idx_band);
 
[fxc, yc] = decimate_for_plot(f_band, Pc_band, MAXPTS_MAIN, true);
[fxn, yn] = decimate_for_plot(f_band, Pn_band, MAXPTS_MAIN, true);
[fxe, ye] = decimate_for_plot(f_band, env_band, MAXPTS_REF,  false);
[fxr, yr] = decimate_for_plot(f_band, ref_band, MAXPTS_REF,  false);
ye = clip2nan(ye, Y_LIM_BAND(1));
yr = clip2nan(yr, Y_LIM_BAND(1));
 
fig = figure('Visible','on','Color','w','Position',[120 120 1100 360],'Renderer',RENDERER);
plot(fxc, yc, 'LineWidth',0.9); hold on;
plot(fxn, yn, 'LineWidth',0.9);
plot(fxe, ye, '--', 'LineWidth',1);
plot(fxr, yr, ':',  'LineWidth',1);
grid on; xlim([-BAND_VIEW_HZ, +BAND_VIEW_HZ]); ylim(Y_LIM_BAND);
xlabel('Frequency [Hz]'); ylabel('Relative PSD [dB]');
title(sprintf('全带频谱（周期图，窄带视图：±%g Hz）', BAND_VIEW_HZ));
legend('相参(周期图)','非相参(周期图)','单脉冲sinc包络(对齐)','相参Dirichlet参考','Location','best');
save_figure(fig, fullfile(OUTDIR,'02_periodogram_pmBAND'), RES_DPI, SAVE_PDF, SAVE_FIG);
 
%% ================= 3) 细节：± zoom_bins × PRF（kHz） ======
f_win    = zoom_bins*PRF;
idx_zoom = abs(f) <= f_win;
fig = figure('Visible','on','Color','w','Position',[180 160 1100 360],'Renderer',RENDERER);
plot(f(idx_zoom)/1e3, Pc_dB(idx_zoom), 'LineWidth',1); hold on;
plot(f(idx_zoom)/1e3, Pn_dB(idx_zoom), 'LineWidth',1);
yl = ylim; stem_k = (-zoom_bins:zoom_bins)*PRF;
for kk = 1:numel(stem_k)
    plot([stem_k(kk) stem_k(kk)]/1e3, yl, 'k:', 'HandleVisibility','off');
end
grid on; xlabel('Frequency [kHz]'); ylabel('Relative PSD [dB]');
title(sprintf('细节：±%d×PRF = ±%.1f kHz', zoom_bins, f_win/1e3));
legend('相参(线谱清晰)','非相参(线谱被抹平)','Location','best');
save_figure(fig, fullfile(OUTDIR,'03_zoom_pmNPRF'), RES_DPI, SAVE_PDF, SAVE_FIG);
 
%% ================= 4) Welch 平滑 PSD（kHz） ===============
[fx_env_w, env_w_plot] = decimate_for_plot(f/1e3, env_dB, MAXPTS_REF, false);
env_w_plot = clip2nan(env_w_plot, Y_LIM_WELCH(1));
fig = figure('Visible','on','Color','w','Position',[240 200 1100 360],'Renderer',RENDERER);
plot(fw/1e3, Pw_c_dB, 'LineWidth',1); hold on;
plot(fw/1e3, Pw_n_dB, 'LineWidth',1);
plot(fx_env_w, env_w_plot, '--', 'LineWidth',1);
grid on; xlim(fs/1e3*[-0.2 0.2]); ylim(Y_LIM_WELCH);
xlabel('Frequency [kHz]'); ylabel('Relative PSD [dB]');
title('Welch 平滑 PSD'); legend('相参 Welch','非相参 Welch','单脉冲sinc包络(对齐)','Location','best');
save_figure(fig, fullfile(OUTDIR,'04_welch'), RES_DPI, SAVE_PDF, SAVE_FIG);
 
fprintf('已在屏幕显示并导出 PNG 到：%s\n', OUTDIR);
 
%% ================= 工具函数 =================
function y = clip2nan(y, yMin)
    y(y<yMin) = NaN;
end
 
function [x2, y2] = decimate_for_plot(x, y, maxPts, keepExtrema)
% 抽点以减少绘图伪影与数据量；keepExtrema=true 每组保 min/max
    if nargin<4, keepExtrema=false; end
    n = numel(x);
    if n<=maxPts, x2=x; y2=y; return; end
    g = ceil(n/maxPts);
    if keepExtrema
        nb = floor(n/g); x2=zeros(2*nb,1); y2=x2; k=0;
        for i=1:nb
            idx=(i-1)*g + (1:g);
            [ymin,imin]=min(y(idx)); [ymax,imax]=max(y(idx));
            k=k+1; x2(k)=x(idx(imin)); y2(k)=ymin;
            k=k+1; x2(k)=x(idx(imax)); y2(k)=ymax;
        end
        [x2,ord]=sort(x2); y2=y2(ord);
    else
        x2 = x(1:g:end); y2 = y(1:g:end);
    end
end
 
function save_figure(fig, base, dpi, save_pdf, save_fig)
% 同时导出 PNG（位图）/可选 PDF（矢量）/可选 FIG（MATLAB）
    exportgraphics(fig, base + ".png", 'Resolution', dpi);
    if save_pdf
        exportgraphics(fig, base + ".pdf", 'ContentType','vector'); %#ok<UNRCH>
    end
    if save_fig
        savefig(fig, base + ".fig");
    end
end