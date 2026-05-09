function res = simulate_comm_link(symTx, bitTx, op, H_comm, snrDb)
%SIMULATE_COMM_LINK 通信链路：发送 -> 物理信道 -> 符号域LMMSE

L = numel(symTx);
noiseVar = 10^(-snrDb/10);

% 发射样本
x = op.tx * symTx;

% 用户接收
noise = sqrt(noiseVar/2) * (randn(L,1) + 1j*randn(L,1));
y = H_comm * x + noise;

% 波形接收变换
z = op.rx * y;

% 等效通信信道
Heff = op.rx * H_comm * op.tx;

% 符号域LMMSE均衡
A = Heff' * Heff + noiseVar * eye(L);
b = Heff' * z;
symHat = A \ b;

% 硬判决与性能
bitHat = qpsk_hard_demod(symHat);
BER = mean(bitHat ~= bitTx);
EVM = sqrt(mean(abs(symHat - symTx).^2) / mean(abs(symTx).^2));

res = struct();
res.txSamples = x;
res.rxSamples = y;
res.rxWaveformDomain = z;
res.Heff = Heff;
res.symHat = symHat;
res.bitHat = bitHat;
res.BER = BER;
res.EVM = EVM;
res.noiseVar = noiseVar;

end
