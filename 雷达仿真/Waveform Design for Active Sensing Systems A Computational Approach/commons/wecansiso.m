function x = wecansiso(N, gamma, x0)
% x = wecan(N, gamma, x0), WeCAN
%   N: length of the sequence
%   gamma: N-by-1, corresponding to weights w_k = gamma_k^2
%   x0: N-by-1, initialization sequence
%   x: N-by-1, the generated sequence

if nargin == 3
    x = x0;
else
    x = exp(1i * 2*pi * rand(N, 1));
end
xPrev = zeros(N,1);
iterDiff = norm(x - xPrev);

gamma(1) = 0;
Gamma = toeplitz(gamma);
eigvalues = eig(Gamma);
gamma(1) = abs(min(eigvalues));
Gamma = toeplitz(gamma)/gamma(1);
C = sqrtm(Gamma);

Alpha = zeros(2*N, N); % Alpha(p,:) is alpha_p
while(iterDiff > 1e-6)
    disp(iterDiff);
    xPrev = x;
    % step 1
    Z = [C.' .* kron(x,ones(1,N)); zeros(N,N)]; % 2N*N
    F = fft(Z); % 2N*N, p(th) row corresponds to alpha_p
    for p = 1:(2*N)
        Alpha(p,:) = sqrt(N) * F(p,:) / norm(F(p,:));
    end
    % step 2
    Nu = ifft(Alpha); % 2N*N
    MuNu = C' .* Nu(1:N,:); % N*N
    for n = 1:N
        x(n) = exp(1i * angle(sum(MuNu(n,:))));
    end
    % stop criterion
    iterDiff = norm(x - xPrev);
end
