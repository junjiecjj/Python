function [x y criterion] = ivcaf_weight(N, g, w, x, y)
% [x y] = ivcaf_weight(N, g, w), ivcaf_weight(N, g, w, x) or ivcaf_weight(N, g, w, x, y)
%   g: (2N-1)-by-N, desired ambiguity function amplitude
%   w: (2N-1)-by-N, weights
%   x: N-by-1, transmitted unimodular, PAR(x) <= rho
%   y: N-by-1, IV filter, energy constrained to be ||y||^2 = N

xPre = zeros(N,1);
yPre = zeros(N,1);
if mod(N,2) ~= 0
    error('N must be even');
end

if nargin <= 4
    y = exp(1i*2*pi*rand(N,1));
    if nargin <= 3
        x = exp(1i*2*pi*rand(N,1));
    end
end

count = 1;
criterion = zeros(300,1);

iterDiff = norm(x - xPre) + norm(y - yPre);
%while (iterDiff>5e-3)
while (count <= 300)
    disp([num2str(count) ': ' num2str(iterDiff)]);
    xPre = x;
    yPre = y;
    % phi
    Phi = zeros(2*N-1, N);
    for k = (-N+1):(N-1)
        for p = (-N/2):(N/2-1)            
            if k>=0
                Phi(k+N,p+N/2+1) = angle((x(1:N-k).*exp(1i*2*pi*p/N*(1:N-k)'))' * y(k+1:N));
            else
                Phi(k+N,p+N/2+1) = angle((x(1-k:N).*exp(1i*2*pi*p/N*(1-k:N)'))' * y(1:N+k));
            end
            Phi(k+N,p+N/2+1) = Phi(k+N,p+N/2+1) + pi*p/N;
        end
    end
    % x
    B = zeros(N,N);
    D1 = zeros(N,N);
    for k = (-N+1):(N-1)
        for p = (-N/2):(N/2-1)
            if k>=0
                J = sparse(diag(exp(-1i*2*pi*p/N*(1:N-k)), k));
            else
                J = sparse(diag(exp(-1i*2*pi*p/N*(1-k:N)), k));
            end
            J = J * exp(1i*pi*p/N) * sinc(p/N);
            B = B + w(k+N,p+N/2+1)*g(k+N,p+N/2+1) * exp(-1i*Phi(k+N,p+N/2+1)) * J;
            D1 = D1 + w(k+N,p+N/2+1) * J * y * y' * J';
        end
    end
    x = inv(D1) * (B*y);
    % y
    D2 = zeros(N,N);
    for k = (-N+1):(N-1)
        for p = (-N/2):(N/2-1)
            if k>=0
                J = sparse(diag(exp(-1i*2*pi*p/N*(1:N-k)), k));
            else
                J = sparse(diag(exp(-1i*2*pi*p/N*(1-k:N)), k));
            end
            J = J * exp(1i*pi*p/N) * sinc(p/N);
            D2 = D2 + w(k+N,p+N/2+1) * J' * x * x' * J;
        end
    end
    y = inv(D2) * (B'*x);

    criterion(count) = sum(sum(w.*(g.^2))) + real(y'*D2*y) - 2*real(y'*B'*x);
    count = count + 1;
    
    iterDiff = norm(x - xPre) + norm(y - yPre);
end
% figure;
% semilogx(1:(count-1), criterion, 'bx-');
% xlabel('No. of Iterations');
% ylabel('Criterion');
% myboldify;