function R = helperMMSECovariance_pd(normalizedPos, P_des, theta_grid, Pt)
M = length(normalizedPos);
epsilon = 1e-6 * Pt / M;
P_des = P_des(:);
theta_grid = theta_grid(:);
G = zeros(length(theta_grid), M * M);
for idxTheta = 1:length(theta_grid)
    theta = theta_grid(idxTheta);
    a = exp(1j * 2 * pi * normalizedPos(:) * sind(theta));
    A = a * a';
    G(idxTheta, :) = reshape(A.', 1, []);
end
cvx_begin quiet
    variable R(M, M) hermitian
    variable alpha1 nonnegative
    expression P(length(theta_grid))
    for idxTheta = 1:length(theta_grid)
        P(idxTheta) = real(G(idxTheta, :) * reshape(R.', M * M, 1));
    end
    minimize(norm(P - alpha1 * P_des, 2))
    subject to
        R >= epsilon * eye(M);
        diag(R) == (Pt / M) * ones(M, 1);
cvx_end
R = (R + R') / 2;
R = Pt * R / real(trace(R));
end