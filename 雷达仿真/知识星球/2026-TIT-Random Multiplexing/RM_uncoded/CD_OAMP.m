%% CD-OAMP
% ------------------------------------------------------------------
% Model:
% y = H * U * s + n, n ~ CN(0, v_n)
% U is a unitary modulation matrix
% **size(H) should be large (typically >100)**
% ------------------------------------------------------------------
% Input:
% (1) H: channel matrix  
% (2) V: right singular vectors of H
% (3) s, signal s (used only when computing MSE)
%     y, v_n: received signal y, noise variance v_n
% (4) dia: singular values of H
% (5) it: maximum number of iterations
% (6) info: struct, see "Denoiser.m" for details
% (7) mod_info: struct, see "Modulations.m" for details
% ------------------------------------------------------------------
% Output:
% (1) MSE, Var: MSE and variance of s_p in iterations (help debug)
% (2) s_p: soft estimate of s
function [MSE, Var, s_p] = CD_OAMP(H, V, s, y, dia, v_n, it, info, mod_info)
    MSE = zeros(1, it);
    Var = zeros(1, it);
    M = length(y);
    N = length(s);
    u_nle = info.mean .* ones(N, 1);
    u_nle = Modulations(u_nle, mod_info, 0);
    v_nle = info.var;
    AHy = H' * y;
    thres_0 = 1e-7;

    % iterations
    for t = 1 : it
        % LE
        [u_le_p, v_le_p] = LE_OAMP(V, AHy, u_nle, v_nle, dia, v_n, M, N);
        [u_le, v_le] = Orth(u_le_p, v_le_p, u_nle, v_nle);
        % cross domain
        u_le = Modulations(u_le, mod_info, 1);
        % NLE
        [u_nle_p, v_nle_p] = Denoiser(u_le, v_le, info);
        MSE(t) = (u_nle_p - s)' * (u_nle_p - s) / N;
        Var(t) = v_nle_p;
        if Var(t) < thres_0
            s_p = u_nle_p;
            MSE(t:end) = max(MSE(t), thres_0);
            Var(t:end) = max(Var(t), thres_0);
            break
        elseif t > 1 && MSE(t) > MSE(t-1)
            MSE(t:end) = MSE(t-1);
            Var(t:end) = Var(t-1);
            break
        end
        s_p = u_nle_p;
        [u_nle, v_nle] = Orth(u_nle_p, v_nle_p, u_le, v_le);
        % cross domain 
        u_nle = Modulations(u_nle, mod_info, 0);
    end
end

%% Orthogonalization
function [u_orth, v_orth] = Orth(u_post, v_post, u_pri, v_pri)
    v_orth = 1 / (1 / v_post - 1 / v_pri);
    u_orth = v_orth * (u_post / v_post - u_pri / v_pri);  
end

%% LE
function [u_post, v_post] = LE_OAMP(V, AHy, u, v, dia, v_n, M, N)
    rho = v_n / v;
    Dia = [dia.^2; zeros(N-M, 1)];
    D = 1 ./ (rho + Dia);
    u_post = V * (D .* (V' * (AHy + rho * u)));
    v_post = v_n / N * sum(D);
end