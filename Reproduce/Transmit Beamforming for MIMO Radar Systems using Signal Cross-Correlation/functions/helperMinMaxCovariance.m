


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


    MM = numel(ang);
    N = size(elPos, 2);
    % Initial covariance is random. x_ is a vector of all elements that are
    % in the upper triangular part of the matrix above the main diagonal.
    x_ = initialCovariance(N);
    % Normalized the desired beam pattern such that the total transmit power is
    % equal to the number of array elements N.
    Pdesired = N * Pdesired / (2*pi*trapz(deg2rad(ang), Pdesired.*cosd(ang))); % Eq.(16)
    Pdesired = Pdesired * 4 * pi;  % Eq.(15)
    % Matrix of steering vectors corresponding to angles in ang
    A = steervec(elPos, [ang; zeros(size(ang))]);
    % Parameters of the barrier method
    
    epsilon = 1e-2;
    stopCriteriaMet = false;

    M1 = MM/4;
    Theta0 = ang(sort(randperm(MM, M1)));
    while ~stopCriteriaMet
    end
 


    R = constrainedCovariance(x, N);
end

























