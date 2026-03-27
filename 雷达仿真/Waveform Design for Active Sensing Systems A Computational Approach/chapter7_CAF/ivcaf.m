function [x y criterion] = ivcaf(N, g, x, y)
% [x y] = ivcaf(N, g), ivcaf(N, g, x) or ivcaf(N, g, x, y)
%   g: (2N-1)-by-N, desired cross ambiguity function amplitude
%   x: N-by-1, transmitted unimodular sequence
%   y: N-by-1, IV filter, energy constrained to be ||y||^2 = N

xPre = zeros(N,1);
yPre = zeros(N,1);
if mod(N,2) ~= 0
    error('N must be even');
end

if nargin <= 3
    y = exp(1i*2*pi*rand(N,1));
    if nargin <= 2
        x = exp(1i*2*pi*rand(N,1));
    end
end

count = 1;
criterion = zeros(500,1);

iterDiff = norm(x - xPre) + norm(y - yPre);
%while (iterDiff>5e-3)
while (count <= 500)
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
        end
    end
    % x
    B = zeros(N, N);
    for k = (-N+1):(N-1)
        for p = (-N/2):(N/2-1)
            if k>=0
                J = sparse(diag(exp(-1i*2*pi*p/N*(1:N-k)), k));
            else
                J = sparse(diag(exp(-1i*2*pi*p/N*(1-k:N)), k));
            end
            B = B + g(k+N,p+N/2+1) * exp(-1i*Phi(k+N,p+N/2+1)) * J;
        end
    end
    x = exp(1i * angle(B*y));   
%     x = B*y;
%     x = sqrt(N) * x / norm(x);
    % y
    ybar = B' * x;
    y = sqrt(N) * ybar / norm(ybar);    

    criterion(count) = 2*N^3 - 2*real(x'*B*y);
    count = count + 1;
    
    iterDiff = norm(x - xPre) + norm(y - yPre);
end
% figure;
% semilogx(1:(count-1), criterion, 'bx-');
% xlabel('No. of Iterations');
% ylabel('Criterion');
% myboldify;