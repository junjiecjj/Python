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