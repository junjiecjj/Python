function MMF = mmf_weight(r, weight)
% r: N*1, correlation, r(0) -- r(N-1)
% weight: (N-1)*1

r = abs(r);
MMF = r(1)^2 / (2 * weight' * (r(2:end).^2));
