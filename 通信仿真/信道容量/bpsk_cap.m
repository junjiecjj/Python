function cap = bpsk_cap(snr_list)
tic;
n = 1e6;



sigma2 = 10.^(-snr_list ./ 10);


w = randn(n, numel(snr_list));
y = 1 + w .* sqrt(sigma2);
p0 = (1 + exp(-2 .* y ./ sigma2)) .^ -1;
p1 = (1 + exp( 2 .* y ./ sigma2)) .^ -1;
mut = 1 + p0.*log2(p0) + p1.*log2(p1);


cap = mean(mut);

toc;