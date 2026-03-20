%% MMSE Denoiser
% ------------------------------------------------------------------
% Model: r = x + n, n ~ N(0, v) or CN(0, v), x ~ P_x
% Goal: find MMSE estimate of x
% ------------------------------------------------------------------
% Input:
% (1) r: vector
% (2) v: positive number
% (3) info: struct with the filed 'type'
% If type is 'BG' or 'BCG', "info" must include 'p_1', 'u_g', 'v_g'
% If type is 'RD', "info" must include fileds 'X', 'P'
% ------------------------------------------------------------------
% Output:
% (1) u_p: mmse estimate of x
% (2) v_p: variance
% ------------------------------------------------------------------
function [u_p, v_p] = Denoiser(r, v, info)
    if ~isreal(v)
        warning('The variance v is not real!')
    elseif v < 0
        warning('The variance v is not positive!')
    end
    type = info.type;
    if strcmpi(type, 'BPSK')
        if ~isreal(r)
            r = real(r);
            v = v / 2;
        end
        [u_p, v_p] = Demod_BPSK(r, v);
    elseif strcmpi(type, 'QPSK')
        [u_1, v_1] = Demod_BPSK(sqrt(2)*real(r), v);    % real part
        [u_2, v_2] = Demod_BPSK(sqrt(2)*imag(r), v);    % imaginary part
        u_p = (u_1 + u_2 * 1i) / sqrt(2);
        v_p = (v_1 + v_2) / 2;
    elseif strcmpi(type, '16QAM')
        X = [-3, -1, 1, 3] / sqrt(10);
        P = [0.25, 0.25, 0.25, 0.25];
        [u_1, v_1] = Demod_RD(real(r), v/2, X, P);    % real part
        [u_2, v_2] = Demod_RD(imag(r), v/2, X, P);    % imaginary part
        u_p = u_1 + u_2 * 1i;
        v_p = v_1 + v_2;
    elseif strcmpi(type, 'BG')
        if ~isreal(r)
            r = real(r);
            v = v / 2;
        end
        p_1 = info.p_1;
        u_g = info.u_g;
        v_g = info.v_g;
        [u_p, v_p] = Demod_BG(r, v, p_1, u_g, v_g);
    elseif strcmpi(type, 'BCG')
        p_1 = info.p_1;
        u_g = info.u_g;
        v_g = info.v_g;
        [u_p, v_p] = Demod_BCG(r, v, p_1, u_g, v_g);
    elseif strcmpi(type, 'RD')
        if ~isreal(r)
            r = real(r);
            v = v / 2;
        end
        X = info.X;
        P = info.P;
        [u_p, v_p] = Demod_RD(r, v, X, P);
    else
        error('The prior distribution is not supported currently!')
    end
end

%% BPSK
function [u_p, v_p] = Demod_BPSK(r, v)
    Exp_lim = 50;
    d = -2 * r / v;
    d = min(max(d, -Exp_lim), Exp_lim);
    p_1 = 1 ./ (1 + exp(d));
    u_p = 2 .* p_1 - 1;
    v_p = mean(1 - u_p.^2);
end

%% Real discrete distribution
% X = [x_1, ..., x_n], Pr(x = x_i) = p_i, P = [p_1, ..., p_n]
function [u_p, v_p] = Demod_RD(r, v, X, P)
    r = r(:);  
    X = X(:)';  
    P = P(:)';
    Exp_lim = 50;
    n = length(X);
    N = length(r);
    p_p = zeros(N, n);
    %
    X2 = X.^2;
    for ii = 1 : n
        x_i = X(ii);
        tmp = x_i^2 - X2;
        tmp = repmat(tmp, N, 1);
        d = (2 * r * (X - x_i) + tmp) / (2*v);
        d = min(max(d, -Exp_lim), Exp_lim);
        p_p(:, ii) = P(ii) ./ sum(P.*exp(d), 2);
    end
    u_p = sum(p_p.*X, 2);
    v_p = sum(p_p.*X2, 2) - u_p.^2;
    v_p = mean(v_p);
end

%% Bernoulli-Gaussian / Bernoulli-Complex Gaussian
% x = b * g, b ~ Bern(p_1), g ~ N(u_g, v_g) or CN(u_g, v_g)
function [u_p, v_p] = Demod_BG(r, v, P, u_g, v_g)
    N = length(r);
    EXP_B = 50;
    u_g = u_g * ones(N, 1);
    % post Bernoull
    c = sqrt((v + v_g) / v);
    d = 0.5 * ((r - u_g).^2 / (v + v_g) - (r.^2) / v);
    d(d > EXP_B) = EXP_B;
    d(d < -EXP_B) = -EXP_B;
    p1 = P ./ (P + (1-P) * c * exp(d));
    % post Gaussian
    v_pg = 1 / (1 / v + 1 / v_g);
    u_pg = v_pg * (r / v + u_g / v_g);
    % post u and v
    u_p = p1 .* u_pg;
    v_p = (p1 - p1.^2) .* (u_pg.^2) + p1 * v_pg;
    v_p = mean(v_p);
end

% Bernoulli-Complex Gaussian
function [u_p, v_p] = Demod_BCG(r, v, P, u_g, v_g)
    N = length(r);
    EXP_B = 50;
    u_g = u_g * ones(N, 1);
    % post Bernoull
    c = (v + v_g) / v;
    d = abs(r - u_g).^2 / (v + v_g) - abs(r).^2 / v;
    d(d > EXP_B) = EXP_B;
    d(d < -EXP_B) = -EXP_B;
    p1 = P ./ (P + (1-P) * c * exp(d));
    % post Gaussian
    v_pg = 1 / (1 / v + 1 / v_g);
    u_pg = v_pg * (r / v + u_g / v_g);
    % post u and v
    u_p = p1 .* u_pg;
    v_p = (p1 - p1.^2) .* (abs(u_pg).^2) + p1 * v_pg;
    v_p = mean(v_p);
end