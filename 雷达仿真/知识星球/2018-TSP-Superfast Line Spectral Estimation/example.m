% For reproducibility
rng(1);

% Parameters
N = 2^7;
K = round(N/10);
index = (1:N)';
snr = 20;	% dB

% Select frequencies by rejection sampling
% (avoid closely located frequencies)
while true
	freqs = sort(rand(K,1));
	if all(diff(freqs)>2/N) ...
			&& (freqs(1)-freqs(end)+1) > 2/N
		break;
	end
end

% Generate signal
alpha = randn(K, 2) * [1;1j];
alpha = alpha./abs(alpha);	% Normalize to magnitude 1
x = exp(-1j*2*pi*index*freqs') * alpha;
noiseVar = mean(abs(x).^2) / 10^(snr/10);
y = x + sqrt(noiseVar/2) * randn(N, 2) * [1;1j];

% Run algorithm
out = superfast_lse(y, index, N, 'verbose',false, 'plot',false);

% Print result
fprintf('True freqs,  Estimated freqs, Difference\n');
[freqs, out.tau, abs(freqs-out.tau)]

