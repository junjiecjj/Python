


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
    Pdesired = N * Pdesired / (2*pi*trapz(deg2rad(ang), Pdesired.*cosd(ang))); % Eq.(16)
    Pdesired = Pdesired * 4 * pi;                                      % Eq.(15)
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

























