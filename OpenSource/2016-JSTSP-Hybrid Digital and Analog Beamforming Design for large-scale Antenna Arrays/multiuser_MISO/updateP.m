

function [P, initPower, lamda] = updateP(Q_t, beta, P_t, K, sigma2)
    lamda = 1;
    while 1
        initPower = 0;
        posi = 0;
        for k = 1:1:K
            tttt = (beta(k)/lamda) - Q_t(k,k)*sigma2;
            if(tttt > 0 )
                initPower = initPower + tttt;
                posi = posi + 1;
            end
        end
        if( abs(initPower / P_t - 1) <= 0.001 )
            disp("find P");
            break;
        end
        if(posi > 0) 
            lamda = lamda + 0.5*(initPower - P_t)/posi;
        else 
            lamda = lamda/4;
        end
    end
    % 求出P
    P = zeros(K,K);
    for kk = 1:1:K
        P(kk,kk) = max([(beta(kk)/lamda) - Q_t(kk,kk)*sigma2, 0]) / Q_t(kk,kk);
    end
end