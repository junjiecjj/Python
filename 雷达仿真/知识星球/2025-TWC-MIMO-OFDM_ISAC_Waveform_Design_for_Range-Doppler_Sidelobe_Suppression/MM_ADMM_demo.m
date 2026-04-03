% This Matlab script can be used to generate the main waveform design results in the paper:
% P. Li, M. Li, R. Liu, Q. Liu, and A. L. Swindlehurst, "MIMO-OFDM ISAC waveform design for range-Doppler sidelobe suppression," IEEE Trans. Wireless Commun., vol. 24, no. 2, pp. 1001-1015, Feb. 2025.
% P. Li, M. Li, R. Liu, Q. Liu, and A. L. Swindlehurst, "Low range-Doppler sidelobe ISAC waveform design: A low-complexity approach," IEEE Trans. Veh. Technol., vol. 74, no. 10, pp. 16544-16549, Oct. 2025.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-26

clear all; clc; close all;
rng('shuffle');

rootDir = fileparts(mfilename('fullpath'));
srcDir  = fullfile(rootDir, 'src');
dataDir = fullfile(rootDir, 'data');
if exist(srcDir, 'dir') ~= 7
    error('src folder not found: %s', srcDir);
end
if exist(dataDir, 'dir') ~= 7
    mkdir(dataDir);
end
addpath(genpath(srcDir));

% Build simulation parameters and one random demo instance.
para = paper_params();
data = build_demo_instance(para);
%% Baseline
fprintf('1. Communication-only baseline\n');
cvx_solver(para.cvx_solver);
[x_comm, info_comm] = comm_wave(data, para);
% Compute ISL / mainlobe / normalized ISL.
[isl_comm, islr_comm, ml_comm] = compute_islr(x_comm, data);
fprintf('[Comm-only] ISL = %.4f dB, ML = %.4f dB, normalized ISL = %.4f dB\n', ...
    10*log10(isl_comm), 10*log10(ml_comm), 10*log10(islr_comm));

fprintf('\n2. Combined-waveform baseline\n');
combined = combined_waveform_baseline(data, para);
fprintf('[Combined] ISL = %.4f dB, ML = %.4f dB, normalized ISL = %.4f dB\n', ...
    combined.isl_db, combined.ml_db, combined.islr_db);

fprintf('\n3. Radar-only baseline\n');
radar = radar_wave(data, para);
fprintf('[Radar-only] ISL = %.4f dB, ML = %.4f dB, normalized ISL = %.4f dB\n', ...
    radar.isl_db, radar.ml_db, radar.islr_db);
%% ISAC waveform design
% ALM-RCG method from the TVT 2025 paper.
fprintf('\n4. ALM-RCG-based ISAC waveform design\n');
ISAC_almrcg = alm_rcg_isac(data, para, x_comm);
fprintf('[ALM-RCG] ISL = %.4f dB, ML = %.4f dB, normalized ISL = %.4f dB\n', ...
    ISAC_almrcg.isl_db, ISAC_almrcg.ml_db, ISAC_almrcg.islr_db);

% MM-ADMM method from the TWC 2025 paper.
fprintf('\n5. MM-ADMM-based ISAC waveform design\n');
ISAC_mmadmm = mm_admm_isac(data, para, x_comm);
fprintf('[MM-ADMM] ISL = %.4f dB, ML = %.4f dB, normalized ISL = %.4f dB\n', ...
    ISAC_mmadmm.isl_db, ISAC_mmadmm.ml_db, ISAC_mmadmm.islr_db);

%% save results
% resultFile = fullfile(dataDir, sprintf('waveform_result_%s.mat', ...
%     datestr(now, 'yyyymmdd_HHMMSS')));
% save(resultFile, 'para', 'data', 'x_comm', 'combined', 'ISAC_almrcg', 'ISAC_mmadmm', 'radar');