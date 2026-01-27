function [DataAll, DataEstAll] = DataGen(Para)
% Generate QPSK and initial distance
% input
%       N_c             # of chirp per frame
%       Para.N_frame         # of frame
% output
%       DataAll         struct, 
%                       QAM_bit         1 × Para.N_frame, QPSK
%                       IM_rng_bit      9 × Para.N_frame, N_s=1024, wherein
%                                       512 is used for IM
%                       IM_vel_bit      5 × Nit, i.e.,
%                                       log2(Nc/Nt=128/4=32)=5bit
%                       QAM             1 × Para.N_frame
%                       IM_rng          1 × Para.N_frame
%                       IM_vel          1 × Para.N_frame
%% DataAll
DataAll = struct( ...
    'QAM_bit',[],...
    'IM_rng_bit',[],...
    'IM_vel_bit',[],...
    ...
    'QAM',[],...
    'IM_rng',[],...
    'IM_vel',[]);

DataAll.QAM_bit = randi([0, 1], 1, 1*Para.N_frame);
DataAll.QAM = ((DataAll.QAM_bit*2-1)*1j + 1)/sqrt(2);
DataAll.QAM(9:10:end) = 1;
% 
DataAll.IM_rng_bit = randi([0, 1], 1, 9*Para.N_frame);
% DataAll.IM_rng_bit(1:9*Para.N_period*Para.N_period_VelStartTxInfo) = 0;
% DataAll.IM_rng_bit = zeros(1, 9*Para.N_frame);
IM_bit_1 = reshape(DataAll.IM_rng_bit, 9, Para.N_frame);
IM_rng_seq = bin2dec(char('0' + IM_bit_1.'));
DataAll.IM_rng = reshape(IM_rng_seq, 1, Para.N_frame);

DataAll.IM_vel_bit = randi([0, 1], 1, 5*Para.N_frame);
DataAll.IM_vel_bit(1:5*Para.N_period*Para.N_period_VelStartTxInfo) = 0;
% DataAll.IM_vel_bit = zeros(1, 5*Para.N_frame);
IM_bit_2 = reshape(DataAll.IM_vel_bit, 5, Para.N_frame);
% IM_bit_2(:,...
%     Para.N_period*Para.N_period_VelStartTxInfo + 1 : end) = repmat(double(dec2bin(30,5).'-48), ...
%     1, Para.N_frame - Para.N_period*Para.N_period_VelStartTxInfo);
IM_vel_seq = bin2dec(char('0' + IM_bit_2.'));
DataAll.IM_vel = reshape(IM_vel_seq, 1, Para.N_frame);
%% DataEstAll
DataEstAll = struct( ...
    'QAM_bit', zeros(1, 1*Para.N_frame),...
    'IM_rng_bit', zeros(1, 9*Para.N_frame),...
    'IM_vel_bit', zeros(1, 5*Para.N_frame),...
    ...
    'QAM', zeros(1, Para.N_frame),...
    'IM_rng', zeros(1, Para.N_frame),...
    'IM_vel', zeros(1, Para.N_frame));
end

