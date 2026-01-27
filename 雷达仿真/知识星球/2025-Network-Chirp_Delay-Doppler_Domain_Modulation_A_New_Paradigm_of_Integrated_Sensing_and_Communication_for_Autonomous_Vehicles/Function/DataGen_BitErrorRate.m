function DataAll = DataGen_BitErrorRate(Nit, NumQAMBit, NumIMRngBit, NumIMVelBit)
% Generate QPSK and initial distance, velocity to embed info
% input
%       Nit             # of iteration
%       NumIMRngBit     # of bits one symbol has
%       NumIMVelBit     # of bits one symbol has
% output
%       DataAll         struct, 
%                       QAM_bit         1 × Nit, QPSK
%                       IM_rng_bit      NumIMBit × Nit, e.g. N_s=1024, wherein
%                                       512 is used for IM
%                       IM_vel_bit      5 × Nit, i.e.,
%                                       log2(Nc/Nt=128/4=32)=5bit
%                       QAM             1 × Nit
%                       IM_rng   1 × Nit
%                       IM_vel   1 × Nit
%% DataAll
DataAll = struct( ...
    'QAM_bit',[],...
    'IM_rng_bit',[],...
    'IM_vel_bit',[],...
    ...
    'QAM',[],...
    'IM_rng',[],...
    'IM_vel',[]);
[symbol, bits] = generateQAM(2^NumQAMBit);
idx = randi(NumQAMBit, Nit, 1);
DataAll.QAM = symbol(idx);
DataAll.QAM_bit = bits(idx, :);

DataAll.IM_rng_bit = randi([0, 1], 1, NumIMRngBit*Nit);
IM_bit_rng = reshape(DataAll.IM_rng_bit, NumIMRngBit, Nit);
IM_seq_rng = bin2dec(char('0' + IM_bit_rng.'));
DataAll.IM_rng = reshape(IM_seq_rng, 1, Nit);

DataAll.IM_vel_bit = randi([0, 1], 1, NumIMVelBit*Nit);
IM_bit_vel = reshape(DataAll.IM_vel_bit, NumIMVelBit, Nit);
IM_seq_vel = bin2dec(char('0' + IM_bit_vel.'));
DataAll.IM_vel = reshape(IM_seq_vel, 1, Nit);
end

