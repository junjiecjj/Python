function X = helperRadComWaveform(H, S, X0, Pt, rho)
% This function implements Riemannian Conjugate Gradient (RCG) algorithm
% described in 
%
% Liu, Fan, Longfei Zhou, Christos Masouros, Ang Li, Wu Luo, and Athina
% Petropulu. "Toward dual-functional radar-communication systems: Optimal
% waveform design." IEEE Transactions on Signal Processing 66, no. 16
% (2018): 4264-4279.
%
% H is a K-by-N channel matrix where K is the number of down-link users
% and N is the number of antennas at the RadCom base station. S is a K-by-L
% symbol matrix, where L is the number of symbols. X0 is a N-by-L matrix of
% waveforms that generate the desired radar beam pattern. Pt is total
% transmit power, and rho is a trade-off parameter to balance the radar and
% communication performance.

delta = 1e-6;
maxNumIterations = 500;
                
[N, M] = size(X0);
S = S * sqrt(Pt/size(S, 1));

A = [sqrt(rho)*H; sqrt(1-rho)*eye(N)];
B = [sqrt(rho)*S; sqrt(1-rho)*X0];

X = (randn(N, M) + 1i*randn(N, M))/sqrt(2);
X = Rx(X, 0, Pt);

% Euclidean gradient
dF = 2 * A' * (A * X - B);

% Projection of the gradient onto the tangent space at X
gradF = Px(X, dF, Pt);

% Descent direction
G = -gradF;

% Armijo rule parameters
s = 1;
beta = 0.5;
sigma = 0.01;

k = 1;
while true
    fx = norm(A*X-B, 'fro').^2;

    % Iterate until the maximum number of iterations is reached or the norm
    % of the gradient is less than delta
    if k > maxNumIterations || norm(gradF, 'fro') < delta
        break
    end

    % If the inner product of the gradient and the descent direction is
    % positive reset the descent direction and make it equal to the
    % negative gradient
    gamma = real(trace(gradF*G'));
    if gamma > 0
        G = -gradF;
        gamma = real(trace(gradF*G'));
    end

    % Step size and new X computation using Armijo rule
    m = 0;
    mu = 1;
    while (-sigma*mu*gamma) > 0
        mu = (beta^m)*s;
        X_ = Rx(X, mu*G, Pt);
    
        fx_ = norm(A*X_ - B, 'fro').^2;

        if (fx(1) - fx_(1)) >= (-sigma*mu*gamma)
            X = X_;
            break;
        end
        m = m + 1;
    end

    % Previous projected gradient
    gradF_ = gradF;

    % Previous descent direction
    G_ = G;

    % Euclidean gradient at X
    dF = 2 * A' * (A * X - B);

    % Projection of the gradient onto the tangent space at X
    gradF = Px(X, dF, Pt);

    % Polak-Ribiere combination coefficient
    tau = real(trace(gradF'* (gradF -  Px(X, gradF_, Pt)))) / real(trace(gradF_'*gradF_));

    % New descent direction is a combination of the current negative
    % gradient and the previous gradient translated into the current
    % tangent space
    G = -gradF + tau *  Px(X, G_, Pt);

    k = k + 1;
end

end

% Tangent space projector
function PxZ = Px(X, Z, Pt)
    [N, M] = size(X);
    d = diag(Z*X');
    PxZ = Z - diag(d)*X*(N/(M*Pt));
end

% Retractor mapping
function RxZ = Rx(X, Z, Pt)
    [N, M] = size(X);
    Y = X + Z;
    d = diag(Y*Y').^(-1/2);
    RxZ = sqrt(M * Pt / N)*diag(d) * Y;
end

