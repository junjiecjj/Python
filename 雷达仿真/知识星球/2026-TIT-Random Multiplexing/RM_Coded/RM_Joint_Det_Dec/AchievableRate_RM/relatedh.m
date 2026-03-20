function A=relatedh(Nr,Ns,rhot,rhor)
% rho = 0.2;   % 相关系数[0,1）越大越相关
    C_R = eye(Nr,Nr);
    for i = 1 : Nr
        for j = 1 : Nr
            if abs(i - j) < 30
                C_R(i, j) = rhor^(abs(i - j));
            end
        end
    end
%     C_R2 = sqrtm(C_R);
    C_R=chol(C_R)';
    C_T = eye(Ns, Ns);
    for i = 1 : Ns
        for j = 1 : Ns
            if abs(i - j) < 30
                C_T(i, j) = rhot^(abs(i - j));
            end
        end
    end

%     C_T2 = sqrtm(C_T);
    C_T=chol(C_T);

    G = sqrt( 1/2 ) * (randn(Nr,Ns) + randn(Nr, Ns)*1i);
    A = C_R * G * C_T;
%     A2 = C_R2 * G * C_T2;
end


