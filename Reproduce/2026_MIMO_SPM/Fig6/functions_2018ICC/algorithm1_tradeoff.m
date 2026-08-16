
function X = algorithm1_tradeoff(H, S, X0, PT, rho)
    epsTol = 1e-10;
    maxIter = 400;
    M = size(H, 2);
    L = size(S, 2);
    I = eye(M);
    Q = rho * (H' * H) + (1 - rho) * I;
    G = rho * H' * S + (1 - rho) * X0;
    Q = (Q + Q') / 2;
    [V, D] = eig(Q);
    lambdaVals = real(diag(D));
    lambdaMin = min(lambdaVals);
    B = V' * G;
    powerTarget = L * PT;
    lambdaLeft = -lambdaMin + 1e-10;
    maxB = max(abs(B(:)));
    lambdaRight = -lambdaMin + sqrt(M / PT) * maxB;
    if lambdaRight <= lambdaLeft
        lambdaRight = lambdaLeft + 1;
    end
    while power_lambda(lambdaRight, lambdaVals, B) > powerTarget
        lambdaRight = 2 * lambdaRight + 1;
    end
    for iter = 1:maxIter
        lambdaMid = (lambdaLeft + lambdaRight) / 2;
        pMid = power_lambda(lambdaMid, lambdaVals, B);
        if abs(pMid - powerTarget) <= epsTol * max(1, powerTarget)
            break;
        end
        if pMid > powerTarget
            lambdaLeft = lambdaMid;
        else
            lambdaRight = lambdaMid;
        end
    end
    lambdaStar = (lambdaLeft + lambdaRight) / 2;
    X = (Q + lambdaStar * I) \ G;
end


function p = power_lambda(lambda, lambdaVals, B)
    den = lambda + lambdaVals;
    p = sum(sum(abs(B).^2 ./ (den.^2)));
end
