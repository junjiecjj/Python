function params = paper_params()
%PAPER_PARAMS Centralized parameters for the RIS-assisted MIMO-FMCW study.
%   Output:
%       params - Struct containing paper parameters, derived linear units,
%                reproduction assumptions, and unit conversion helpers.
%
%   Units:
%       Frequencies are in Hz, time is in seconds, distance is in meters,
%       velocity is in m/s, power fields ending in _W are linear Watts, and
%       power fields ending in _dBm are logarithmic dBm values.

params.paper.title = "RIS-assisted MIMO-FMCW radar NLOS target parameter estimation";
params.paper.originalTitle = "RIS辅助MIMO-FMCW雷达的非视距目标参数估计方法";
params.paper.authors = "Wang Jixuan, Tian Tuanwei, Deng Hao, Ma Rui";

params.array.Nt = 4;
params.array.Nb = 4;
params.array.Nr_default = 16;
params.array.streams = params.array.Nt;

params.radar.c = 3e8;
params.radar.fc = 77e9;
params.radar.bandwidth = 500e6;
params.radar.chirpTime = 50e-6;
params.radar.sampleRate = 2e6;
params.radar.numChirps = 256;
params.radar.numFastTimeSamples = round(params.radar.sampleRate * params.radar.chirpTime);
params.radar.rcs = 1;
params.radar.lambda = params.radar.c / params.radar.fc;
params.radar.slope = params.radar.bandwidth / params.radar.chirpTime;
params.radar.rangeResolution = params.radar.c / (2 * params.radar.bandwidth);
params.radar.velocityResolution = params.radar.lambda / (2 * params.radar.numChirps * params.radar.chirpTime);

params.power.txPower_dBm = 10;
params.power.noisePower_dBm = 10;
params.power.txPower_W = dbm2w_local(params.power.txPower_dBm);
params.power.noisePower_W = dbm2w_local(params.power.noisePower_dBm);

params.channel.ricianK_dB = 10;
params.channel.ricianK_linear = db2pow_local(params.channel.ricianK_dB);
params.channel.pathLossExponent = 2;
params.channel.referenceDistance_m = 1;
params.channel.sourceToRisDistance_m = 15;
params.channel.risToTargetDistance_m = 20;
params.channel.model = "stage2_rician_ris_domain_effective";
params.channel.Hsr_size = [params.array.Nr_default, params.array.Nt];
params.channel.Hrd_size = [params.array.Nr_default, params.array.Nr_default];

params.optim.maxIter = 1000;
params.optim.tolerances = [1e-2, 1e-3, 1e-4];
params.optim.risReflectionAmplitude = 1;

params.targets.range_m = [25, 20, 10, 5];
params.targets.velocity_mps = [-1, 1, -1, 1];

params.repro.rngSeed = 20251009;
params.repro.resetRngInGenerateChannels = true;
params.repro.status = "stage2_base_model";

params.utils.dbm2w = @dbm2w_local;
params.utils.w2dbm = @w2dbm_local;
params.utils.db2pow = @db2pow_local;
params.utils.pow2db = @pow2db_local;
end

function watts = dbm2w_local(dbm)
%DBM2W_LOCAL Convert dBm to Watts.
watts = 1e-3 .* 10.^(dbm ./ 10);
end

function dbm = w2dbm_local(watts)
%W2DBM_LOCAL Convert Watts to dBm.
validateattributes(watts, {'numeric'}, {'positive'});
dbm = 10 .* log10(watts ./ 1e-3);
end

function linearPower = db2pow_local(dbValue)
%DB2POW_LOCAL Convert dB to linear power ratio.
linearPower = 10.^(dbValue ./ 10);
end

function dbValue = pow2db_local(linearPower)
%POW2DB_LOCAL Convert linear power ratio to dB.
validateattributes(linearPower, {'numeric'}, {'positive'});
dbValue = 10 .* log10(linearPower);
end
