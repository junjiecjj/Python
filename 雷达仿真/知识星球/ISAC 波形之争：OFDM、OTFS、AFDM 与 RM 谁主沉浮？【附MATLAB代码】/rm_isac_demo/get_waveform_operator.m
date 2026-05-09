function op = get_waveform_operator(waveformName, params)
%GET_WAVEFORM_OPERATOR 返回统一线性发送/接收算子
%
% 约定：
%   x = U * s
%   z = W * y, 其中 W = U^H
%   H_eff = W * H * U

L = params.L;
M = params.M;
N = params.N;

switch upper(waveformName)
    case 'RM'
        F = unitary_dft_matrix(L);
        rng(params.rmPermutationSeed);
        perm = randperm(L);
        P = eye(L);
        P = P(perm, :);

        U = P * F;
        meta.perm = perm;
        meta.description = 'RM: random permutation + unitary DFT';

    case 'OFDM'
        F = unitary_dft_matrix(L);
        U = F';
        meta.description = 'OFDM: unitary IFFT basis';

    case 'OTFS'
        % 教学型块OTFS算子：等效为 Doppler 维 IDFT 展开
        FN = unitary_dft_matrix(N);
        U = kron(FN', eye(M));
        meta.description = 'OTFS: simplified rectangular-pulse block operator';

    case 'AFDM'
        U = afdm_daft_matrix(L, params.afdm.c1, params.afdm.c2);
        meta.description = 'AFDM: teaching-form DAFT operator';

    otherwise
        error('Unsupported waveform: %s', waveformName);
end

W = U';

op = struct();
op.name = waveformName;
op.tx = U;
op.rx = W;
op.meta = meta;

end
