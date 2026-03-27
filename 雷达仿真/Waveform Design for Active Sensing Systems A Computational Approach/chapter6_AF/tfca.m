function x = tfca(N, deltaF, P, x0)
% tfcan: Time-Frequency CA
%   x = tfca(N, deltaF, P)
%   x is the N-by-1 waveform with good time-frequency properties
%   deltaF is an M-by-1 vector specifying bandwidth-normalized Doppler
%   frequncies, e.g. [0 deltaf 2*deltaf ... (M-1)*deltaf].'
%   P: P-1 is the maximum time delay under consideration
%
%   X = tfca(N, fd, P, x0)
%   x0 is the initialization waveform, N-by-1
%
%   date: 03/08/2009


M = length(deltaF); % 2M-1 grid points on the Doppler axis
if nargin <= 3
    x0 = exp(1i*2*pi * rand(N,1));
end

Rd = N * eye(M);
RdTilde = kron(Rd, eye(P)); % MP-by-MP
RdTildeRoot = sqrtm(RdTilde);

% step 0
x = x0;
xPre = zeros(N, 1);

count = 0;
while (norm(x - xPre, 'fro') > 1e-3)
    count = count + 1;
    disp(norm(x - xPre, 'fro'));
    xPre = x;
    % step 2
    Z = zeros(N+P-1, M*P);
    for m = 1:M
        for p = 1:P
            Z(p:(p+N-1), (m-1)*P+p) = x .* exp(1i*2*pi * (1:N)'...
                * deltaF(m));
        end
    end
    [Ubar S Utilde] = svd(RdTildeRoot * Z', 'econ');
    U = Utilde * Ubar';
    % step 1
    UR = U * RdTildeRoot;
    for n = 1:N
        rho = 0;
        for m = 1:M
            ux = n; uy = (m-1)*P + 1;
            for p = 0:(P-1)
                rho = rho + exp(-1i*2*pi*n*deltaF(m)) * UR(ux+p, uy+p);
            end
        end
        x(n) = exp(1i * angle(rho));
    end
end