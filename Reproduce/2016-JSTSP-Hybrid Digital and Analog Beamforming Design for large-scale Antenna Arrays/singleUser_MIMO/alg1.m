

function Vrf = alg1(F, Ns, gamma2, sigma2, epsilon)
    Nrf = Ns;
    N = size(F, 1);
    Vrf = ones(N, Nrf);
    last_iter_obj = 0.0;
    iter_obj = 0.0;
    diff = 1.0;
    while diff > epsilon
        for j = 1:Nrf
            Vrfj = Vrf(:, [1:j-1, j+1:end]);
            % Compute Cj and Gj as Eq.(13)
            Cj = eye(Nrf-1) + (gamma2/sigma2) * (Vrfj' * F * Vrfj);
            Gj = (gamma2/sigma2) * F - (gamma2/sigma2)^2 * F * Vrfj * pinv(Cj) * Vrfj' * F;
            % Vrf update loop
            for i = 1:N
                eta_ij = 0.0;
                % Sum l != i loop
                for l = setdiff(1:N, i)
                    eta_ij = eta_ij + Gj(i, l) * Vrf(l, j);
                end
                % Value assignment as per (14)
                if eta_ij == 0
                    Vrf(i,j) = 1;
                else
                    Vrf(i,j) = eta_ij / abs(eta_ij);
                end
            end
        % Save the last result
        last_iter_obj = iter_obj;
        % Calculate objective function of (12a)
        iter_obj = log2(det(real(eye(Nrf) + (gamma2/sigma2) * Vrf' * F * Vrf)));
        % Calculate difference of last and current objective function
        diff = abs((iter_obj - last_iter_obj) / iter_obj);
        end
    end
end

