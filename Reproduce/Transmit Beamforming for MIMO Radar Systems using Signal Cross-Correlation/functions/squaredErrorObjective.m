


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




