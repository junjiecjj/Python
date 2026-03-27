function [s w mse msebound] = cyc_freq(N, Gamma, beta, rho, s0)
% [s w mse msebound] = cyc_freq(N, Gamma, rho, s0), freq-domain Lagrange 
% plus cyclic algorithm for probing sequence and receive filter design
%   Gamma: N-by-N, the covariance matrix of interference (noise+jamming)
%   beta: E{|alpha|^2} where alpha is the RCS coefficient
%   rho: (optional) maximum allowed peak-to-average power ratio, in [1 N]
%   s0: (optional) N-by-1, sequence for initialization
%
%   s: N-by-1, the probing sequence
%   w: N-by-1, the receive filter, w = inv(P) * s
%   mse: w'Rw/(|w's|^2) = 1/(s'*inv(R)*s)
%	msebound: (2N-1)/sum(z ./ (beta * z + g)) - beta, the MSE when 
%		the power spectrum (i.e., z) of s is ideal

if nargin < 5
    s0 = exp(1i * 2*pi * rand(N,1));
    if nargin < 4
        rho = N;
    end
end

cn = Gamma(:,1); % covariance of noise+interference
cn = [cn; conj(cn(N:-1:2))]; % (2N-1)-by-1

g = fft(cn) / (2*N-1); % power spectrum of noise+interference, (2N-1)-by-1
g = real(g); % eliminate numerical problems
rh = (beta * N + sum(g)) / (sum(sqrt(g)));

z = (rh * sqrt(g) - g) / beta; % (2N-1)-by-1
if sum(z >= 0) < (2*N-1)
    % binary search for lambda
    left = 0;
    right = 1;
    fval = 0;
    while (abs(fval - N) > 1e-2)
        lambda = (left + right) / 2;
        z = (lambda * rh * sqrt(g) - g) / beta; % (2N-1)-by-1
        fval = sum(max([z.'; zeros(1,2*N-1)]));
        if (fval < N)
            left = lambda;
        elseif (fval > N)
            right = lambda;
        end
    end
    z = max([z.'; zeros(1, 2*N-1)]).';
end

% If the spectrum match is perfect
msebound = (2*N-1) / sum(z ./ (beta*z + g)) - beta;
% disp(['MSE lower bound: ' num2str(msebound)]);

% calculate {|x_p|}
xabs = sqrt(z); % 2N-by-1, {|x_p|}, p=1,...,2N-1

% % examine the optimal sequence/filter spectrum
% habs = xabs ./ (beta * z + g);
% figure;
% plot(linspace(0,1,2*N-1), z, 'r', ...
%     linspace(0,1,2*N-1), habs.^2, 'b');
% legend('optimal s', 'optimal w');
% xlabel('f');
% ylabel('Power Spectrum');
% myboldify;
% drawnow;

% iteration
s = s0;
spre = zeros(N, 1);
iterdiff = norm(s - spre);

while(iterdiff > 1e-3)
    spre = s;
    
    % update x
    x = xabs .* exp(1i * phase(fft(s, 2*N-1))); % (2N-1)-by-1
    
    % update s
    nu = ifft(x) * sqrt(2*N-1); % (2N-1)-by-1
    nu = nu(1:N); % N-by-1    
    if rho == 1
        s = exp(1i * phase(nu));
    elseif rho < N
        s = vectorfitpar(nu, N, rho);
    else
        s = nu / norm(nu) * sqrt(N);
    end
    
    % iteration criterion
    iterdiff = norm(s - spre);
    
    R = Rcalc(s, Gamma, beta);
    mse = 1/real(s' * (R \ s));
    sptrmfit = norm(xabs - abs(fft(s, 2*N-1)/sqrt(2*N-1)))^2;
    disp(['||s-spre|| = ' num2str(iterdiff) '; MSE = ' num2str(mse) ...
       ' || |x| - |F''s| || = ' num2str(sptrmfit)]);
end

R = Rcalc(s, Gamma, beta);
w = R \ s;
mse = 1/real(s' * w);
