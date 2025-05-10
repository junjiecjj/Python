function cap = bpsk_cap_loop(snr_list)
tic
n = 1e6;   % number of test

cap = zeros(size(snr_list)); % allocate output
for i = 1:numel(snr_list)    % iterate over snr_list
  sigma2 = 10.^(-snr_list(i) ./ 10);
  mut_sum = 0;
  for t = 1:n                % Monte-Carlo simul
    w = randn(1);                             % standard Gaussian noise
    y = 1 + w .* sqrt(sigma2);                % received symbol
    p0 = (1 + exp(-2 .* y ./ sigma2)) .^ -1;  % de-modulation: +1
    p1 = (1 + exp( 2 .* y ./ sigma2)) .^ -1;  %                -1
    mut = 1 + p0.*log2(p0) + p1.*log2(p1);    % calc mutal information
    mut_sum = mut_sum + mut;                  % accumulate
  end
  cap(i) = mut_sum / n;                       % output
end
toc;