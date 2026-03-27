function x = pertfca(N, deltaF, P, x0)
% pertfcan: Periodic Time-Frequency CA
%   x = pertfca(N, deltaF, P)
%   x is the N-by-1 waveform with good time-frequency properties
%   deltaF is an M-by-1 vector specifying bandwidth-normalized Doppler
%   frequncies, e.g. [0 deltaf 2*deltaf ... (M-1)*deltaf].'
%   where deltaf = 1/N
%   P: P-1 is the maximum time delay under consideration
%
%   X = pertfca(N, fd, P, x0)
%   x0 is the initialization waveform, N-by-1
%
%   date: 11/15/2009


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
iterDiff = norm(x - xPre, 'fro');
while (iterDiff > 1e-3)
    count = count + 1;
    disp(iterDiff);
    xPre = x;
    % step 2
    Z = zeros(N, M*P);
    for m = 1:M
        for p = 1:P
            Z(:, (m-1)*P+p) = circshift(x .* exp(1i*2*pi*(1:N)'*deltaF(m)), p-1);
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
            if n<=(N-P+1)            
                for p = 0:(P-1)
                    rho = rho + exp(-1i*2*pi*n*deltaF(m)) * UR(ux+p, uy+p);
                end
            else
                p1 = n-(N-P+1);
                for p = 0:(P-1-p1)
                    rho = rho + exp(-1i*2*pi*n*deltaF(m)) * UR(ux+p, uy+p);
                end
                for p = 0:(p1-1)
                    rho = rho + exp(-1i*2*pi*n*deltaF(m)) * ...
                        UR(1+p, uy+(P-1)-(p1-1)+p);
                end
            end
        end
        x(n) = exp(1i * angle(rho));
    end    
    iterDiff = norm(x - xPre, 'fro');
end