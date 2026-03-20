%% OAMP(SE)
function [V_post, V_le, V_nle] = OAMP_SE_qpsk(dia, v_n, it, N)
    V_post = zeros(1, it);
    V_le = zeros(1, it);
    V_nle = zeros(1, it);
    v_nle = 1;
    for i = 1 : it
        % LE
        tmp = v_n ./ dia.^2;
        Dia = 1 ./ (tmp + v_nle);
        v_le_post = v_nle - v_nle^2 * sum(Dia) / N;
        v_le = 1 / (1 / v_le_post - 1 / v_nle);
        V_le(i) = v_le;
        % NLE
        v_nle_post = Dem_SE(v_le);
        V_post(i) = v_nle_post;
        v_nle = 1 / (1 / v_nle_post - 1 / v_le);
        V_nle(i) = v_nle;
    end
end

%% Demodulator (SE)
function v_p = Dem_SE(v)
    upl = 100;
    snr = 1 / v;
    v_p = 1 - integral(@(x) f_QPSK(x, snr), -upl, upl);
end

% 
function y = f_QPSK(x, snr) 
    y = exp(-x.^2/2) .* tanh(snr-sqrt(snr).*x) / sqrt(2*pi);
end
