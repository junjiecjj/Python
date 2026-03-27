function PAF = percafdis(x, y)
% PAF = percafdis(x, y), draw the discrete cross PAF
% PAF(k,p) = \sum_{n=1}^N x(n) x'(n-k MOD N) exp(-j 2pi (n-k)p/N)
% Here (a MOD b) = ((a-1) mod b) + 1 where mod is the normal mod operator
%   x, y: N-by-1
%   PAF: (2N-1)-by-N

if nargin == 1
    y = x;
end

N = length(x);
PAF = zeros(2*N-1, N);

for k = (-N+1):(N-1)
    for p = (-N/2):(N/2-1)
        rkp = 0;
        for n = 1:N
            rkp = rkp + y(n) * conj(x(mod(n-k-1,N) + 1)) * ...
                exp(-1i * 2*pi * (n-k) * p / N);
        end
        PAF(k+N, p+N/2+1) = rkp;
    end
end

PAF = abs(PAF);

figure;
PAFplot = PAF / max(PAF(:));
imagesc(-N+1:N-1,-N/2:N/2-1, 20*log10(PAFplot'),[-40 0]);
colormap(flipud(colormap('hot')));
set(gca,'YDir', 'normal');
xlabel('k'); ylabel('p');
if (norm(x-y) < 1e-6) % AF, not CAF
    title('Discrete-PAF $\tilde{r}(k,p)$', 'Interpreter', 'LaTex');
else
    title('Discrete-cross-PAF $\tilde{r}_{xy}(k,p)$', 'Interpreter', 'LaTex');
end
colorbar;
myboldify;