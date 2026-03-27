function x = wecansisostoppar(rho, N, gamma, NFull, stopIndex, lambda, x0)
% x = wecansisostoppar(rho, N, gamma, NFull, stopIndex, lambda, x0)
% Stopband SISO WeCAN with PAR constraint
%   rho: PAR(x) = max{|x(n)|^2} <= rho (average power is 1)
%   N: the length of the sequence x
%   gamma: N-by-1, corresponding to weights w_k = gamma_k^2
%   NFull: the length of x after padding zeros in the tail
%   stopIndex: an NFull-by-1 binary vector corresponding to NFull FFT
%       frequencies. 1 corresponds to a frequency that should be avoided
%   lambda: 0<lambda<1 controls relative weighting: lambda for stopband
%       suppression and 1-lambda for range sidelobe suppression
%   x0: N-by-1, the initialization sequence
%   x: N-by-1, the generated sequence

if nargin == 7
    x = x0;
else
    x = exp(1i * 2*pi * rand(N,1));
end

gamma(1) = 0;
Gamma = toeplitz(gamma); % N-by-N
eigvalues = eig(Gamma);
gamma(1) = abs(min(eigvalues));
Gamma = toeplitz(gamma)/gamma(1); % N-by-N
C = sqrtm(Gamma); % N-by-N, C'C = Gamma

Alpha = zeros(2*N, N); % Alpha(p,:) is alpha_p
stopIndex = logical(stopIndex);

xPre = zeros(N,1);
iterDiff = norm(x - xPre);

while (iterDiff > 1e-3)
    disp(['iterDiff = ' num2str(iterDiff)]);
    xPre = x;
    % minimize w.r.t {alpha_p}
    Z = [C.' .* kron(x,ones(1,N)); zeros(N,N)]; % 2N-by-N
    F = fft(Z); % 2N-by-N, p(th) row corresponds to beta_p
    for p = 1:(2*N)
        Alpha(p,:) = sqrt(N) * F(p,:) / norm(F(p,:));
    end
    % minimize w.r.t x
    w = 1/sqrt(NFull) * fft(x, NFull); % NFull-by-1
    w(stopIndex) = 0;
    c1 = sqrt(NFull) * ifft(w); % NFull-by-1
    
    Nu = ifft(Alpha); % 2N-by-N
    MuNu = C' .* Nu(1:N,:); % N-by-N
    c2 = sum(MuNu, 2); % N-by-1
    
    xGoal = lambda * c1(1:N) + (1-lambda) * c2;
    
    if rho == 1
        x(1:N) = exp(1i * angle(xGoal));
    else
        x(1:N) = vectorfitpar(xGoal, N, rho); % N-by-1
    end

    % stop criterion
    iterDiff = norm(x - xPre);
end