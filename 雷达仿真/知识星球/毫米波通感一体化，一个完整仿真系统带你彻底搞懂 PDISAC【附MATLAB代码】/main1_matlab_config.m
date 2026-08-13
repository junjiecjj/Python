% MATLAB_CONFIG
% Single-file MATLAB configuration for the PDISAC simulator. Run as a script;
% it defines the raw configuration, validates it, derives every constant, and
% unpacks the legacy uppercase workspace variables the pipeline expects.
%
% Populates:
%   PDISAC_cfg        raw configuration struct (system/scene/sensing/...)
%   PDISAC_repo_root  workspace root (parent of simulation_matlab/)
%   derived constants fc, c, Lambda, B, fs, noise powers, gains, waveform
%                     timing, transmit power, scene ranges, CFAR settings
%   profile handles   dataset_*_cfg, *_statistics_cfg, inference_cfg
%
% This is the MATLAB-side counterpart of the Python ML configuration in
% alg_pdisac/configs/pdisac_system.json. Edit the raw values below; all
% derived quantities update automatically.

%% ===== Raw configuration ====================================================
PDISAC_cfg = struct();
PDISAC_cfg.schema_version = 1;
PDISAC_cfg.active_profile = 'demo';

% ---- System / RF ----
PDISAC_cfg.system.carrier_hz              = 77e9;
PDISAC_cfg.system.bandwidth_hz            = 150e6;
PDISAC_cfg.system.temperature_k           = 290.0;
PDISAC_cfg.system.noise_figure_com_db     = 3.0;
PDISAC_cfg.system.noise_figure_sensing_db = 3.0;
PDISAC_cfg.system.gain_tx_db              = 60.0;
PDISAC_cfg.system.gain_rx_db              = 60.0;
PDISAC_cfg.system.gain_ue_db              = 50.0;
PDISAC_cfg.system.tx_position_m           = [0.0; 0.0; 0.0];
PDISAC_cfg.system.rx_position_m           = [0.0; 0.0; 0.0];
PDISAC_cfg.system.tx_antennas             = 1;
PDISAC_cfg.system.rx_antennas             = 1;
PDISAC_cfg.system.ue_antennas             = 1;

% ---- Waveform ----
PDISAC_cfg.waveform.chips             = 512;
PDISAC_cfg.waveform.blocks            = 256;
PDISAC_cfg.waveform.symbols_per_block = 8;

% ---- Scene ----
PDISAC_cfg.scene.roi_m                 = [0.0 100.0; -100.0 100.0; 0.0 0.0];
PDISAC_cfg.scene.ue_position_m         = [55.0; -60.0; 0.0];
PDISAC_cfg.scene.ue_velocity_mps       = [-10.0; -10.0; 0.0];
PDISAC_cfg.scene.target_positions_m    = [60.0 70.0 90.0 50.0 20.0; ...
                                          -25.0 15.0 30.0 40.0 30.0; ...
                                          0.0 0.0 0.0 0.0 0.0];
PDISAC_cfg.scene.target_velocities_mps = [-15.0 20.0 0.0 3.0 0.0; ...
                                          12.0 -10.0 25.0 -10.0 20.0; ...
                                          0.0 0.0 0.0 0.0 0.0];
PDISAC_cfg.scene.target_rcs_range      = [10.0 20.0];
PDISAC_cfg.scene.scatterer_rcs_range   = [5.0 10.0];
PDISAC_cfg.scene.scatterers            = 200;
PDISAC_cfg.scene.communication_reflectors = 10;
PDISAC_cfg.scene.max_ue_speed_mps      = 20.0;
PDISAC_cfg.scene.max_target_speed_mps  = 30.0;

% ---- Sensing / CFAR ----
PDISAC_cfg.sensing.snr_db              = 10.0;
PDISAC_cfg.sensing.zero_static_doppler = true;
PDISAC_cfg.sensing.cfar.guard_range             = 2;
PDISAC_cfg.sensing.cfar.guard_doppler           = 2;
PDISAC_cfg.sensing.cfar.training_range          = 4;
PDISAC_cfg.sensing.cfar.training_doppler        = 4;
PDISAC_cfg.sensing.cfar.false_alarm_probability = 1e-4;
PDISAC_cfg.sensing.cfar.peak_select             = false;

% ---- RDPDNet dataset-generation profiles ----
PDISAC_cfg.pdisac_dataset.dataset_train = struct( ...
    'system_override', struct('gain_tx_db', 45.0, 'gain_rx_db', 50.0, 'gain_ue_db', 42.0), ...
    'random_scenes', 100, 'monte_carlo', 1, ...
    'symbols_per_block', [4 8 16 32 64 128], ...
    'snr_db', [-10 -5 0 5 10 15 20 25 30], ...
    'fixed_targets', 5, 'scatterers', 200, ...
    'output_db', 'alg_pdisac/data/dataset_train.db');

PDISAC_cfg.pdisac_dataset.dataset_test = struct( ...
    'system_override', struct('gain_tx_db', 45.0, 'gain_rx_db', 50.0, 'gain_ue_db', 42.0), ...
    'random_scenes', 100, 'monte_carlo', 1, ...
    'symbols_per_block', [4 8 16 32 64 128], ...
    'snr_db', [-10 -5 0 5 10 15 20 25 30], ...
    'fixed_targets', 5, 'scatterers', 200, ...
    'output_db', 'alg_pdisac/data/dataset_test.db');

PDISAC_cfg.pdisac_dataset.dataset_smoke = struct( ...
    'system_override', struct('gain_tx_db', 45.0, 'gain_rx_db', 50.0, 'gain_ue_db', 42.0), ...
    'random_scenes', 2, 'monte_carlo', 1, ...
    'symbols_per_block', 4, ...
    'snr_db', [0 20], ...
    'fixed_targets', 5, 'scatterers', 200, ...
    'output_db', 'alg_pdisac/data/dataset_smoke.db');

PDISAC_cfg.pdisac_dataset.dataset_inference = struct( ...
    'input_mat', 'simulation_matlab/exported_statistics/dataset_inference_input.mat', ...
    'output_mat', 'simulation_matlab/exported_statistics/dataset_inference_output.mat');

% ---- Analysis-statistics sweeps ----
PDISAC_cfg.statistics.sensing = struct( ...
    'system_override', struct('noise_figure_com_db', 1.0, 'gain_tx_db', 45.0, ...
                              'gain_rx_db', 50.0, 'gain_ue_db', 30.0), ...
    'scene_override', struct( ...
        'ue_position_m', [30.0; 70.0; 0.0], ...
        'ue_velocity_mps', [10.0; 10.0; 0.0], ...
        'target_velocities_mps', [-15.0 20.0 0.0 3.0 0.0; ...
                                  12.0 -10.0 25.0 3.0 20.0; ...
                                  0.0 0.0 0.0 0.0 0.0]), ...
    'random_scenes', 10, 'monte_carlo', 100, ...
    'symbols_per_block', [4 8 16 32 64 128], ...
    'snr_db', [-10 -5 0 5 10 15 20 25 30], ...
    'fixed_targets', 5, 'scatterers', 200, ...
    'output_file', 'simulation_matlab/exported_statistics/statistics_sensing.mat');

PDISAC_cfg.statistics.communication = struct( ...
    'system_override', struct('gain_tx_db', 45.0, 'gain_rx_db', 50.0, 'gain_ue_db', 42.0), ...
    'random_scenes', 1, 'monte_carlo', 10, ...
    'symbols_per_block', [4 8 16 32 64 128], ...
    'snr_db', [-20 -15 -10 -5 0 5 10 15 20], ...
    'fixed_targets', 5, 'scatterers', 200, ...
    'output_csv', 'simulation_matlab/exported_statistics/statistics_communication.csv', ...
    'rician_k_db', 6.0);

PDISAC_cfg.statistics.communication_v3 = struct( ...
    'system_override', struct('gain_tx_db', 45.0, 'gain_rx_db', 50.0, 'gain_ue_db', 42.0), ...
    'random_scenes', 10, 'monte_carlo', 10, ...
    'symbols_per_block', [4 8 16 32 64 128], ...
    'snr_db', [-20 -15 -10 -5 0 5 10 15 20], ...
    'fixed_targets', 5, 'scatterers', 200, ...
    'output_csv', 'simulation_matlab/exported_statistics/statistics_communication_v3.csv', ...
    'rician_k_db', 6.0);

PDISAC_cfg.statistics.channel = struct( ...
    'system_override', struct('gain_tx_db', 45.0, 'gain_rx_db', 50.0, 'gain_ue_db', 42.0), ...
    'random_scenes', 1, 'monte_carlo', 200, ...
    'symbols_per_block', [4 8 16 32 64 128], ...
    'snr_db', 0, ...
    'output_file', 'simulation_matlab/exported_statistics/statistics_channel.mat');

% ---- RDPDNet inference (MATLAB -> Python bridge) ----
PDISAC_cfg.inference.checkpoint         = 'alg_pdisac/exps/pdnet_afm/checkpoints/best_checkpoint.pth';
PDISAC_cfg.inference.device             = 'auto';
PDISAC_cfg.inference.batch_size         = 8;
% Absolute path to the conda env "paper_isac" interpreter. An absolute path is
% used directly by inference_run_pdnet (repo-relative paths are joined to the
% repo root instead). Change here if your conda prefix differs.
PDISAC_cfg.inference.python_executable  = '/Users/rysheng/miniconda3/envs/afm/bin/python';
PDISAC_cfg.inference.exchange_directory = '';

%% ===== Validation ==========================================================
if bitand(PDISAC_cfg.waveform.chips, PDISAC_cfg.waveform.chips - 1) ~= 0
    error("PDISAC:InvalidConfig", "waveform.chips must be a power of two.");
end
if mod(PDISAC_cfg.waveform.chips, 2 * PDISAC_cfg.waveform.symbols_per_block) ~= 0
    error("PDISAC:InvalidConfig", "chips must be divisible by 2*symbols_per_block.");
end

%% ===== Repository root ======================================================
PDISAC_repo_root = string(fileparts(fileparts(mfilename("fullpath"))));

%% ===== Derived constants (unpacked into the workspace) =====================
% ---- Physical / RF ----
c      = physconst('LightSpeed');
fc     = PDISAC_cfg.system.carrier_hz;
Lambda = c / fc;
B      = PDISAC_cfg.system.bandwidth_hz;
fs     = B;

% ---- Noise ----
N_F_com_db = PDISAC_cfg.system.noise_figure_com_db;
N_F_sen_db = PDISAC_cfg.system.noise_figure_sensing_db;
N_F_com    = 10^(N_F_com_db / 10);
N_F_sen    = 10^(N_F_sen_db / 10);
T_ref      = PDISAC_cfg.system.temperature_k;
Noise_power_com = physconst('Boltzmann') * T_ref * N_F_com * B;
Noise_power_sen = physconst('Boltzmann') * T_ref * N_F_sen * B;

% ---- Gains ----
G_tx_db = PDISAC_cfg.system.gain_tx_db;
G_rx_db = PDISAC_cfg.system.gain_rx_db;
G_ue_db = PDISAC_cfg.system.gain_ue_db;
G_tx    = 10^(G_tx_db / 10);
G_rx    = 10^(G_rx_db / 10);
G_ue    = 10^(G_ue_db / 10);

% ---- Array geometry ----
N_tx_ant    = PDISAC_cfg.system.tx_antennas;
N_rx_ant    = PDISAC_cfg.system.rx_antennas;
N_ue_ant    = PDISAC_cfg.system.ue_antennas;
Tx_position = PDISAC_cfg.system.tx_position_m(:);
Rx_position = PDISAC_cfg.system.rx_position_m(:);

% ---- Waveform / timing ----
N_chip          = PDISAC_cfg.waveform.chips;
N_block         = PDISAC_cfg.waveform.blocks;
N_sym_per_block = PDISAC_cfg.waveform.symbols_per_block;
T_chip          = 1 / B;
T_pmcw          = N_chip * T_chip;
L_slot_per_sym  = N_chip / (2 * N_sym_per_block);
N_slot          = N_chip / L_slot_per_sym;

% ---- Transmit power from sensing SNR ----
SNR_Tx_db = PDISAC_cfg.sensing.snr_db;
SNR_Tx    = 10.^(SNR_Tx_db / 10);
P_tx      = Noise_power_sen .* SNR_Tx;

% ---- Scene ----
Region_of_interest = PDISAC_cfg.scene.roi_m;
VEL_RANGE_UE       = PDISAC_cfg.scene.max_ue_speed_mps;
VEL_RANGE_TAR      = PDISAC_cfg.scene.max_target_speed_mps;
N_ref_tars         = PDISAC_cfg.scene.communication_reflectors;

% ---- CFAR ----
N_guard_range   = PDISAC_cfg.sensing.cfar.guard_range;
N_guard_doppler = PDISAC_cfg.sensing.cfar.guard_doppler;
N_train_range   = PDISAC_cfg.sensing.cfar.training_range;
N_train_doppler = PDISAC_cfg.sensing.cfar.training_doppler;
P_fa            = PDISAC_cfg.sensing.cfar.false_alarm_probability;
peak_select     = PDISAC_cfg.sensing.cfar.peak_select;

%% ===== Profile handles =====================================================
dataset_train_cfg     = PDISAC_cfg.pdisac_dataset.dataset_train;
dataset_test_cfg      = PDISAC_cfg.pdisac_dataset.dataset_test;
dataset_smoke_cfg     = PDISAC_cfg.pdisac_dataset.dataset_smoke;
dataset_inference_cfg = PDISAC_cfg.pdisac_dataset.dataset_inference;
sensing_statistics_cfg          = PDISAC_cfg.statistics.sensing;
communication_statistics_cfg    = PDISAC_cfg.statistics.communication;
communication_v3_statistics_cfg = PDISAC_cfg.statistics.communication_v3;
channel_statistics_cfg          = PDISAC_cfg.statistics.channel;
inference_cfg                   = PDISAC_cfg.inference;
