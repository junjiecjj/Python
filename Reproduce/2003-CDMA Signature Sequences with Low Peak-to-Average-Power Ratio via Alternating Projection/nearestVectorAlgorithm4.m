


function s = nearestVectorAlgorithm4(z, c, rho)
    % Algorithm 4 from Tropp et al., "Designing structured tight frames via alternating projection"
    % Solves: min ||s - z||^2  s.t. ||s||^2 = c, PAR(s) <= rho
    % Input:
    %   z   : complex column vector of length d
    %   c   : desired squared norm
    %   rho : PAR bound (1 <= rho <= d)
    % Output:
    %   s   : optimal vector (if feasible)

    d = length(z);
    if rho < 1 || rho > d
        error('rho must satisfy 1 <= rho <= d');
    end
    delta = sqrt(c * rho / d);
    % Normalize z to unit norm (algorithm assumes)
    z = z / norm(z);
    a = abs(z);
    [a_sorted, idx] = sort(a);     % ascending order
    % Precompute cumulative squares for efficiency
    cum_sq = cumsum(a_sorted.^2);
    cum_sq_full = [0; cum_sq];     % cum_sq_full(i+1) = sum of first i squares

    for k = 0:d
        n_rest = d - k;            % number of retained components
        if n_rest == 0
            % All components truncated
            if abs(c - d*delta^2) < 1e-12
                s = delta * exp(1j * angle(z));
                return;
            else
                continue;
            end
        end
        % Uniqueness check: only when we have both truncated and retained components
        if k > 0 && k < d && a_sorted(n_rest) == a_sorted(n_rest+1)
            continue;
        end
        rest_idx = idx(1:n_rest);   % indices of the (d-k) smallest entries
        % Special case: all retained entries have zero magnitude
        if all(a(rest_idx) == 0)
            if k*delta^2 > c
                continue;
            end
            const = sqrt((c - k*delta^2) / n_rest);
            % Feasibility check: constant must not exceed delta to satisfy PAR
            if const > delta + 1e-12
                continue;
            end
            s = zeros(d,1);
            s(rest_idx) = const * exp(1j * angle(z(rest_idx)));
            s(idx(n_rest+1:end)) = delta * exp(1j * angle(z(idx(n_rest+1:end))));
            return;
        end
        % Normal case
        sum_rest = cum_sq_full(n_rest+1);   % sum of squares of retained entries
        need = c - k*delta^2;
        if need < -1e-12
            continue;
        end
        gamma = sqrt(need / sum_rest);
        if any(gamma * a(rest_idx) > delta + 1e-12)
            continue;
        end
        % Construct solution
        s = zeros(d,1);
        s(rest_idx) = gamma * z(rest_idx);
        s(idx(n_rest+1:end)) = delta * exp(1j * angle(z(idx(n_rest+1:end))));
        return;
    end

    error('Algorithm 4: No feasible solution exists. Check that 1 <= rho <= d and c > 0.');
end