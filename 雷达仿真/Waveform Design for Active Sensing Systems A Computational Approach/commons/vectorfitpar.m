function s = vectorfitpar(z, power, rho)
% s = vectorfitpar(z, power, rho)
% Tropp's Alternating Projection algorithm to compute the nearest vector
% under the PAR (peak-to-average power ratio) constraint
% PAR(s) = max{|s(n)|^2}/(power/N) <= rho
%   z: N-by-1 vector, could be complex-valued
%   power: the squared norm of s, i.e. ||s||^2
%   rho: the PAR of s <= rho
%   s: N-by-1 vector, obtained by sovling the following min problem:
%       min_s || s-z ||
%       s.t. PAR(s) <= rho
%            ||s||^2 = power

if rho == 1
    s = exp(1i * phase(z));
    s = s/norm(s) * sqrt(power);
    return;
end

N = length(z);
delta = sqrt(power*rho/N); % max{|s(n)}} <= delta
[zsort index] = sort(abs(z), 'ascend');
s = zeros(N,1);

for k = N:-1:1 % {k components of z with smallest magnitude} denoted as II
    ind = index(1:k);
    if ~any(z(ind)) % if all elements in II are zero
        s(ind) = sqrt((power - (N-k) * delta^2) / k);
        break;
    else
        gamma = sqrt((power - (N-k) * delta^2) / sum((abs(z(ind))).^2));
        sTmp = gamma * z(ind);
        if all((abs(sTmp)-(1e-7)) <= delta) % satisfying the power constraint
            % 1e-7 is introduced to prevent numerical errors
            s(ind) = sTmp;
            break;
        end
    end
end
% Besides II, the other N-k components
ind = index(k+1:N);
s(ind) = delta * exp(1i * angle(z(ind)));
        