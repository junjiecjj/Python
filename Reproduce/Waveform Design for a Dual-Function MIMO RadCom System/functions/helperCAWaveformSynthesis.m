




function X = helperCAWaveformSynthesis(R, M, rho)
    % This function generates a set of waveforms that form the desired
    % covariance matrix R. M is the number of waveform samples and rho is a
    % parameter controlling the resulting peak-to-average power ratio (PAR).
    % This algorithm is described in the chapter 14 of
    %
    % He, Hao, Jian Li, and Petre Stoica. Waveform design for active sensing
    % systems: a computational approach. Cambridge University Press, 2012.
    %
    % and in
    %
    % Stoica, Petre, Jian Li, and Xumin Zhu. "Waveform synthesis for
    % diversity-based transmit beampattern design." IEEE Transactions on Signal
    % Processing 56, no. 6 (2008): 2593-2598.
    N = size(R, 1);
    epsilon = 0.1;
    % P can control the autocorrelation level for each waveform. If P = M the
    % optimization is trying to minimize sidelobes for across all M-1 lags.
    P = M;
    % Arbitrary semi-unitary matrix
    U = (randn(P + M - 1, N*P) + 1i*randn(P + M - 1, N*P))/sqrt(2);
    % Expended covariance is used to minimize the autocorrelation sidelobes. See
    % eq. (14.10) and (14.11)
    Rexp = kron(R, eye(P));
    % Square root of the expended covariance
    [L, D] = ldl(Rexp);
    Rexp_sr = sqrt(D)*L';
    % Cyclic algorithm
    numIter = 0;
    maxNumIter = 1000;
    while true
        % Z has waveforms that form R but do not satisfy the PAR constraint
        % Eq. (14.6)
        Z = sqrt(M) * U * Rexp_sr;
        X = zeros(M, N);
        Xexp = zeros(P + M - 1, N*P);
        % Go through each waveform
        for n = 1:N
            % Retrieve the nth waveform from the expended matrix Z
            zn = getzn(Z, M, P, n);
            gamma = R(n, n);
            % Enforces PAR constraint using nearest vector algorithm 
            xn = nearestVector(zn, gamma, rho);
            X(:, n) = xn;
            % Compute new expended waveform matrix
            for p = 1:P
                Xexp(p:p+M-1, (n-1)*P + p) = xn;
            end
        end
        % Recompute U. Eq. (5) and (11) in the paper
        [Ubar, ~, Utilde] = svd(sqrt(M)*Rexp_sr*Xexp', 'econ');
        U_ = Utilde * Ubar';
    
        numIter = numIter + 1;
        if norm(U-U_) < epsilon || numIter > maxNumIter
            break;
        else
            U = U_;
        end
    end
    X = X';
end






