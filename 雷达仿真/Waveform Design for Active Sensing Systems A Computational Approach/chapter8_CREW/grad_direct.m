function [s w mse] = grad_direct(N, Gamma, beta, s0)
% [s w mse] = grad_direct(N, Gamma, rho, s0), gradient method (BFGS) for
% unimodular probing sequence and receive filter design
%   Gamma: N-by-N, the covariance matrix of interference (noise+jamming)
%   beta: E{|alpha|^2} where alpha is the RCS coefficient
%   s0: (optional) N-by-1, sequence for initialization
%
%   s: N-by-1, the probing sequence
%   w: N-by-1, the receive filter, w = inv(R) * s
%   mse: w'Rw/(|w's|^2) = 1/(s'*inv(R)*s)

if nargin < 4
    phi = 1i * 2*pi * rand(N,1);
else
    phi = phase(s0);
end

% Display = 'iter', 'notify', 'off'
options = optimset('LargeScale', 'off', 'Display','off', ...
     'GradObj', 'on', 'TolFun', 1e-3, 'MaxIter', 200);

[phi, fval] = fminunc(@core_criterion, phi, options);
s = exp(1i * phi);

mse = 1/(-fval);
mse = real(mse); % get rid of 0.0000i
R = Rcalc(s, Gamma, beta);
w = R \ s; % filter

    % Below is a nested function so that the extra parameters "N", "beta"
    % and "Gamma" can be used by core_criterion in fminunc
    function [f g] = core_criterion(phi)
        % [f g] = core_criterion(phi).
        % The CORE criterion f = -s' * inv(R) * s where s = exp(1i * phi)
        % (This R does not include the zero-delay s.) MSE = 1/(-f)
        % g is the gradient of f
        %   phi: N-by-1
        %   f: value of the criterion
        %   g: N-by-1, gradient
        
        s = exp(1i * phi);
        
        % construct all shifted versions of s
        % N-by-(2N-1)
        
        % s(N) s(N-1) ... s(1)   0    ...  0
        %  0    s(N)      s(2)  s(1)       0
        %  .     0          .   s(2)       .
        %  .                .              .
        %  0     0        s(N) s(N-1) ... s(1)
        
        S = toeplitz([s(N); zeros(N-1,1)], [s(end:-1:1).' zeros(1,N-1)]);
        
        % compute R
        R = beta * (S * S' - s * s') + Gamma; % N-by-N
        
        % compute f
        % Rinv = inv(R);
        f = -s' * (R \ s); % f = -s' * Rinv * s;
        
        if nargout == 2
            % compute g
            g = zeros(N,1);
            for k = 1:N
                
                % compute partial R w.r.t. phi_k
                S1 = S(:, N-(k-1):(2*N-k)); % N-by-N
                S2 = diag(1i*s(k) * ones(N,1)); % N-by-N diagonal
                dR = S1 * S2' + S2 * S1';                
                sk = [zeros(k-1,1); 1i*s(k); zeros(N-k,1)]; % N-by-1
                dR = dR - s * sk' - sk * s';                
%                 for p = -(k-1):(N-k)
%                     if p~=0
%                         sp = S(:,p+N);
%                         spp = [zeros(k-1+p,1); 1i*s(k); zeros(N-k-p,1)];
%                         dR = dR + spp * sp' + sp * spp';
%                     end
%                 end
                dR = beta * dR;
                
                % compute g(k)                
                dRinv = -R \ (dR / R); % dRinv = -Rinv * dR * Rinv;
                Rinv_s = R \ s; % Rinv_s = Rinv * s;
                g(k) = 2 * real(-1i*conj(s(k)) * Rinv_s(k)) + ...
                    real(s' * dRinv * s);
                % the second 'real' is for numerical reasons
            end
            g = -g;
        end
    end

end