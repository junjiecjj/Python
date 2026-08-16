


function X = RiemannianConjugateGradient(H, S, X0, Pt, rho)
% RiemannianConjugateGradient  Solve dual-functional waveform design
%   with per-antenna power constraint using Riemannian conjugate gradient.
%   这是按照自己理解实现的RCG算法
%   X = RiemannianConjugateGradient(H, S, X0, Pt, rho)
%
%   Inputs:
%       H   - K-by-N channel matrix (K users, N antennas)
%       S   - K-by-L symbol matrix (L symbols)
%       X0  - N-by-L reference waveform matrix (from radar-only design)
%       Pt  - total transmit power
%       rho - trade-off parameter (0 <= rho <= 1)
%   Output:
%       X   - optimized N-by-L waveform matrix satisfying per-antenna
%             power constraint: diag(X*X') = (L*Pt/N)*ones(N,1)
%
%   This implementation follows Algorithm 2 in:
%       Liu, Zhou, Masouros, Li, Luo, Petropulu, 
%       "Toward Dual-functional Radar-Communication Systems: 
%        Optimal Waveform Design", IEEE TSP, 2018.
%
%   Modified from an open-source implementation with corrected tangent
%   space projection (added real() to satisfy the orthogonality condition).

    % -------------------------------------------------------------------------
    % Retraction mapping for oblique manifold.
    % Maps a point X + Z back to the manifold where each row has norm
    % sqrt(M * Pt / N).
    % -------------------------------------------------------------------------
    function RxZ = Rx(X, Z, Pt)
        [N, M] = size(X);
        Y = X + Z;
        d = diag(Y * Y') .^ (-1/2);
        RxZ = sqrt(M * Pt / N) * diag(d) * Y;
    end
    
    % -------------------------------------------------------------------------
    % Projection onto the tangent space of the oblique manifold at X.
    % Corrected: takes real part of diag(Z*X') to enforce the orthogonality
    % condition Re( diag(Z*X') ) = 0.
    % -------------------------------------------------------------------------
    function PxZ = Px(X, Z, Pt)
        [N, M] = size(X);
        d = real(diag(Z * X'));      % CRITICAL: real part only
        PxZ = Z - diag(d) * X * (N / (M * Pt));
    end

    delta = 0.5e-6;
    maxNumIterations = 600;

    [N, M] = size(X0);
    % Scale S to ensure correct energy scaling (optional)
    S = S * sqrt(Pt / size(S, 1));

    % Build auxiliary matrices A and B
    A = [sqrt(rho) * H; sqrt(1-rho) * eye(N)];
    B = [sqrt(rho) * S; sqrt(1-rho) * X0];

    % Initialize random feasible point on oblique manifold
    X = (randn(N, M) + 1i * randn(N, M)) / sqrt(2);
    X = Rx(X, 0, Pt);   % retraction to meet row norm constraint

    % Euclidean gradient
    dF = 2 * A' * (A * X - B);
    % Project to tangent space: Riemannian gradient
    gradF = Px(X, dF, Pt);
    % Descent direction (negative Riemannian gradient)
    G = -gradF;

    % Armijo line search parameters
    s = 1;
    beta = 0.5;
    sigma = 0.01;
    k = 1;

    while true
        fx = norm(A * X - B, 'fro')^2;

        if k > maxNumIterations || norm(gradF, 'fro') < delta
            break;
        end

        % Reset direction if gradient and descent direction are not descending
        gamma = real(trace(gradF * G'));
        if gamma > 0
            G = -gradF;
            gamma = real(trace(gradF * G'));
        end

        % Armijo line search
        m = 0;
        mu = 1;
        while (-sigma*mu*gamma) > 0
            mu = (beta^m) * s;
            X_new = Rx(X, mu * G, Pt);
            fx_new = norm(A * X_new - B, 'fro')^2;
            if fx - fx_new >= -sigma * mu * gamma
                X = X_new;
                break;
            end
            m = m + 1;
        end

        % Store previous gradient and direction
        gradF_prev = gradF;
        G_prev = G;

        % Update Euclidean gradient at new X
        dF = 2 * A' * (A * X - B);
        gradF = Px(X, dF, Pt);

        % Polak-Ribière combination coefficient
        tau_num = real(trace(gradF' * (gradF - Px(X, gradF_prev, Pt))));
        tau_den = real(trace(gradF_prev' * gradF_prev));
        if tau_den == 0
            tau = 0;
        else
            tau = tau_num / tau_den;
        end

        % New conjugate direction
        G = -gradF + tau * Px(X, G_prev, Pt);

        k = k + 1;
    end
end

