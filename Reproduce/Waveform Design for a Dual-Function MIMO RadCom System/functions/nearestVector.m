



% Nearest vector algorithm, pages 70-71
function z = nearestVector(z, gamma, rho)
    M = numel(z);
    S = z'*z;
    z = sqrt(M*gamma/S)*z;

    beta = gamma * rho;

    if all(abs(z).^2 <= beta)
        return;
    else
        ind = true(M, 1);
        for i = 1:M
            [~, j] = max(abs(z).^2);

            z(j) = sqrt(beta)*exp(1i*angle(z(j)));

            ind(j) = false;
            S = z(ind)'*z(ind);
            z(ind) = sqrt((M-i*rho)*gamma/S)*z(ind);

            if all(abs(z(ind)).^2 <= (beta+eps))
                return;
            end
        end
    end

end



