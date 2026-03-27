function Jp = sarJJ(N, P, p)
% sarJJ: return a shifting matrix used in SAR imaging
%   Jp = sarJJ(N, p)
%   Jp is an (N+P-1)-by-(N+P-1) matrix with N 1's on the upper p'th
%   off-diagonal, p is in [0 P-1]

if abs(p) >= P
    error('p must be between -(P-1) and P-1');
end

Jp = diag(ones(N+P-1-abs(p), 1), p);
% Jp = zeros(N+P-1, N+P-1);
% px = 0; py = p;
% for n = 1:N
%     px = px + 1; py = py + 1;
%     Jp(px, py) = 1;
% end