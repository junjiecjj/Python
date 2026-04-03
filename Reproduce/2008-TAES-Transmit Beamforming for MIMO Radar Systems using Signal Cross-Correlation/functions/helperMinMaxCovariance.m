


function R = helperMinMaxCovariance(elPos, Pdesired, ang)
    % This function computes a waveform covariance matrix that generates a
    % desired transmit beam pattern. The computation is based on the  Maximum Error Optimization
    % described in 
    %
    % Fuhrmann, Daniel R., and Geoffrey San Antonio. "Transmit beamforming for
    % MIMO radar systems using signal cross-correlation." IEEE Transactions on
    % Aerospace and Electronic Systems 44, no. 1 (2008): 171-186.
    %
    % elPos is a 3-by-N matrix of array element positions normalized by the
    % wavelength. Pdesired is the desired beam pattern evaluated at the angles
    % specified in ang.

    % elPos = normalizedPos;
    % Pdesired = Bdes;
    % 

    MM = numel(ang);
    N = size(elPos, 2); 
    x_ = initialCovariance(N); 
    Pdesired = N * Pdesired / (2*pi*trapz(deg2rad(ang), Pdesired.*cosd(ang))); % Eq.(16)
    Pdesired = Pdesired * 4 * pi;  % Eq.(15)
    A = steervec(elPos, [ang; zeros(size(ang))]);
    
    epsilon = 1e-4;
    stopCriteriaMet = false;

    M1 = floor(MM/2);
    Theta0idx = sort(randperm(MM, M1));
    Theta0 = ang(Theta0idx);
    Ptheta0 = Pdesired(Theta0idx);
    % Matrix of steering vectors corresponding to angles in ang
    
    lmax = 10;
    z0 = inf;
    it = 0;
    maxIter = 1e2;
    while ~stopCriteriaMet & it < maxIter
        it = it + 1
        [x, z] = SDPOptimization(elPos, Theta0, Ptheta0);
        abs(z - z0)
        if abs(z - z0) < epsilon
            stopCriteriaMet = true;
        end
        z0 = z;
        F = constrainedCovariance(x, N);
        P_ = real(diag(A'*F*A).');
        % Squared error weighted by angle
        Err = abs(Pdesired - P_);
        [~, indices] = maxk(Err, lmax);
        Theta0idx = union(Theta0idx, indices);

        Theta0 = ang(Theta0idx);
        Ptheta0 = Pdesired(Theta0idx);
    end
 
    R = constrainedCovariance(x, N);
end

























