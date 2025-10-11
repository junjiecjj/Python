function R = helperMMSECovariance(elPos, Pdesired, ang)
% This function computes a waveform covariance matrix that generates a
% desired transmit beam pattern. The computation is based on the squared
% error optimization described in 
%
% Fuhrmann, Daniel R., and Geoffrey San Antonio. "Transmit beamforming for
% MIMO radar systems using signal cross-correlation." IEEE Transactions on
% Aerospace and Electronic Systems 44, no. 1 (2008): 171-186.
%
% elPos is a 3-by-N matrix of array element positions normalized by the
% wavelength. Pdesired is the desired beam pattern evaluated at the angles
% specified in ang.

N = size(elPos, 2);

% Initial covariance is random. x_ is a vector of all elements that are
% in the upper triangular part of the matrix above the main diagonal.
x_ = initialCovariance(N);

% Normalized the desired beam pattern such that the total transmit power is
% equal to the number of array elements N.
Pdesired = N * Pdesired / (2*pi*trapz(deg2rad(ang), Pdesired.*cosd(ang)));
Pdesired = Pdesired * 4 * pi;

% Matrix of steering vectors corresponding to angles in ang
A = steervec(elPos, [ang; zeros(size(ang))]);

% Parameters of the barrier method
mu = 4;

% The barrier term is weighted by 1/t. At each iteration t is multiplied
% by mu to decrease the contribution of the barrier function.
t = 0.02;
epsilon = 1e-1;
stopCriteriaMet = false;

J_ = squaredErrorObjective(x_, t, A, Pdesired, ang);
while ~stopCriteriaMet
    % Run Newton optimization using x_ as a starting point
    x = newtonOptimization(x_, t, A, Pdesired, ang);

    J = squaredErrorObjective(x, t, A, Pdesired, ang);
    
    if abs(J) < abs(J_)
        stopCriteriaMet = abs(J - J_) < epsilon;
        x_ = x;
        J_ = J;
    else
        % Increased t by too much, step back a little.
        t = t / mu;
        mu = max(mu * 0.8, 1.01);
    end

    t = t * mu;
end

R = constrainedCovariance(x, N);
end

function x = newtonOptimization(x_, t, A, Pdesired, ang)
    epsilon = 1e-3;

    % Parameters for Armijo rule
    s = 2;
    beta = 0.5;
    sigma = 0.1;
    
    stopCriteriaMet = false;
    J_ = squaredErrorObjective(x_, t, A, Pdesired, ang);
    
    while ~stopCriteriaMet
        [g, H] = gradientAndHessian(x_, t, A, Pdesired, ang);

        % Descent direction
        d = -(H\g);

        % Compute step size and the new value x using the Armijo rule
        m = 0;
        gamma = g'*d;
        while true 
            mu = (beta^m)*s;
            x = x_ + mu*d;
    
            J = squaredErrorObjective(x, t, A, Pdesired, ang);

            if abs(J_) - abs(J) >= (-sigma*mu*gamma)
                x_ = x;
                stopCriteriaMet = abs(J - J_) < epsilon;
                J_ = J;
                break;
            end
            m = m + 1;
        end
    end
end

function [G, H] = gradientAndHessian(x, t, A, Pdesired, ang)
    N = size(A, 1);
    M = N*(N-1);
    
    F = constrainedCovariance(x, N);

    numAng = numel(ang);

    FinvFi = zeros(N, N, M);
    Alpha = zeros(M, numAng);

    idxs = find(triu(ones(N, N), 1));
    [r, c] = ind2sub([N N], idxs);

    for i = 1:M/2
        [Fi_re, Fi_im] = basisMatrix(N, r(i), c(i));

        % Matrix inverses used in Eq. (26) and (27)
        FinvFi(:, :, i) = F\Fi_re;
        FinvFi(:, :, i + M/2) = F\Fi_im;

        % Eq. (29)
        Alpha(i, :) = 2*real(conj(A(r(i), :)) .* A(c(i), :));
        Alpha(i + M/2, :) = -2*imag(conj(A(r(i), :)) .* A(c(i), :));
    end

    G = zeros(M, 1);
    H = zeros(M, M);

    D = (real(diag(A'*F*A).') - Pdesired) .* cosd(ang);

    ang_rad = deg2rad(ang);

    for i = 1:M
        % Eq. (33a)
        G(i) = -trace(squeeze(FinvFi(:, :, i))) * (1/t) + 2*trapz(ang_rad, Alpha(i, :) .* D);

        for j = i:M
            % Eq. (33b)
            H(i, j) = trace(squeeze(FinvFi(:, :, i))*squeeze(FinvFi(:, :, j))) * (1/t) ...
                + 2*trapz(ang_rad, Alpha(i, :).*Alpha(j, :).*cosd(ang));
        end
    end

    H = H + triu(H, 1)';
end

function [Fi_re, Fi_im] = basisMatrix(N, i, j)
    Fi_re = zeros(N, N);
    Fi_re(i, j) = 1;
    Fi_re(j, i) = 1;

    Fi_im = zeros(N, N);
    Fi_im(i, j) = 1i;
    Fi_im(j, i) = -1i;
end

function J = squaredErrorObjective(x, t, A, Pdesired, ang)
    % Squared error between the desired beam pattern in Pdesired and the
    % beam pattern formed by a covariance matrix defined by the vector x
    % containing the above diagonal elements
    N = size(A, 1);
    
    % Beam patter defined by x
    F = constrainedCovariance(x, N);
    P_ = real(diag(A'*F*A).');

    % Squared error weighted by angle
    E = abs(Pdesired - P_).^2 .* cosd(ang);

    % Total error over all angles
    J = trapz(deg2rad(ang), E);

    % Barrier function
    d = eig(F);
    if all(d >= 0)
        phi = -log(prod(d));
    else
        phi = Inf;
    end 

    J = J + (1/t)*phi;
end

function F = constrainedCovariance(x, N)
    % Reconstruct the covariance matrix from a vector x of above diagonal
    % values. The diagonal elements are all equal to 1.
    Re = zeros(N, N);
    Im = zeros(N, N);
    M = numel(x);

    idxs = triu(ones(N, N), 1) == 1;
    Re(idxs) = x(1:M/2);
    Im(idxs) = x(M/2+1:end);

    F = eye(N, N);
    F = F + Re + Re.' + 1i*Im - 1i*Im.';
end

function x = initialCovariance(N)
    % Create a random covariance matrix
    X = randn(N, 10*N) + 1i*randn(N, 10*N);
    L = sum(conj(X) .* X, 2);

    X = X./sqrt(L);
    R = X*X';

    M = N*(N-1);
    x = zeros(M, 1);

    % Select the elements that are above the main diagonal
    idxs = triu(ones(N, N), 1) == 1;
    x(1:M/2) = real(R(idxs));
    x(M/2+1:end) = imag(R(idxs));
end