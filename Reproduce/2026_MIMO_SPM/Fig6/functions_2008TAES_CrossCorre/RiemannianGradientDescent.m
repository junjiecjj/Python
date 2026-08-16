

function X = RiemannianGradientDescent(H, S, X0, Pt, rho)
    % RiemannianGradientDescent  Solve dual-functional waveform design
    %   with per-antenna power constraint using Riemannian gradient descent.
    %
    %   X = RiemannianGradientDescent(H, S, X0, Pt, rho)
    %   Inputs:
    %       H   - K-by-N channel matrix (K users, N antennas)
    %       S   - K-by-L symbol matrix (L symbols)
    %       X0  - N-by-L reference waveform matrix (from radar-only design)
    %       Pt  - total transmit power
    %       rho - trade-off parameter (0 <= rho <= 1)
    %   Output:
    %       X   - optimized N-by-L waveform matrix satisfying per-antenna
    %             power constraint: diag(X*X') = (L*Pt/N)*ones(N,1)
    %   This implementation is a simplified version of the RCG algorithm
    %   (Algorithm 2 in Liu et al., TSP 2018), using only the negative
    %   Riemannian gradient as descent direction.
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
    % Enforces Re( diag(Z*X') ) = 0.
    % -------------------------------------------------------------------------
    function PxZ = Px(X, Z, Pt)
        [N, M] = size(X);
        d = real(diag(Z * X'));
        PxZ = Z - diag(d) * X * (N / (M * Pt));
    end

    delta = 0.5e-6;               % Gradient norm tolerance
    maxNumIterations = 600;       % Maximum iterations

    [N, M] = size(X0);
    S = S * sqrt(Pt / size(S, 1)); % Normalize symbol energy

    % Build auxiliary matrices A and B
    A = [sqrt(rho) * H; sqrt(1-rho) * eye(N)];
    B = [sqrt(rho) * S; sqrt(1-rho) * X0];

    % Initialize random feasible point on oblique manifold
    X = (randn(N, M) + 1i * randn(N, M)) / sqrt(2);
    X = Rx(X, 0, Pt);              % Retraction to enforce row norms

    % Armijo line search parameters
    s = 1;          % initial step size
    beta = 0.5;     % reduction factor
    sigma = 0.01;   % sufficient decrease parameter

    k = 1;
    while true
        % Compute Euclidean gradient
        dF = 2 * A' * (A * X - B);
        % Project to tangent space -> Riemannian gradient
        gradF = Px(X, dF, Pt);
        fx = norm(A * X - B, 'fro')^2;
        % Check termination
        if k > maxNumIterations || norm(gradF, 'fro') < delta
            break;
        end
        % Descent direction = negative Riemannian gradient
        G = -gradF;
        gamma = real(trace(gradF * G'));  % = -||gradF||^2 < 0
        % Armijo line search to find step size mu
        m = 0;
        mu = 1;
        while true
            mu = (beta^m) * s;
            X_new = Rx(X, mu * G, Pt);
            fx_new = norm(A * X_new - B, 'fro')^2;
            if fx - fx_new >= -sigma * mu * gamma
                X = X_new;
                break;
            end
            m = m + 1;
        end
        k = k + 1;
    end
end

