function AF = cafdis(x, y)
% AF = cafdis(x, y), draw the discrete cross AF
% AF(k,p) = \sum_{n=1}^N y(n) x*(n-k) exp(-j2\pi (n-k)p/N)
%   x, y: N-by-1
%   AF: (2N-1)-by-N

if nargin == 1
    y = x;
end

N = length(x);
AF = zeros(2*N-1, N);
for k = (-N+1):(N-1)
    for p = (-N/2):(N/2-1)
        if k>=0
            AF(k+N,p+N/2+1) = (x(1:N-k).*exp(1i*2*pi*p/N*(1:N-k)'))' * y(k+1:N);
        else
            AF(k+N,p+N/2+1) = (x(1-k:N).*exp(1i*2*pi*p/N*(1-k:N)'))' * y(1:N+k);
        end
    end
end
AF = abs(AF);
% Saturation
% AF(AF > AF(N,N/2+1)) = AF(N,N/2+1);
% Saturation

figure;
AFplot = AF / max(AF(:));
imagesc(-N+1:N-1,-N/2:N/2-1, 20*log10(AFplot'),[-40 0]);
%imagesc(-N+1:N-1,-N/2:N/2-1, AF');
colormap(flipud(colormap('hot')));
set(gca,'YDir', 'normal');
xlabel('k'); ylabel('p');
if (norm(x-y) < 1e-6) % AF, not CAF
    title('Discrete-AF $\bar{r}(k,p)$', 'Interpreter', 'LaTex');
else
    title('Discrete-CAF $\bar{r}_{xy}(k,p)$', 'Interpreter', 'LaTex');
end
colorbar;
myboldify;