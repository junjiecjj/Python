function A=relatedh(Nr,Ns,rho)
% rho = 0.2;   % 相关系数[0,1）越大越相关
    C_R = eye(Nr,Nr);
    for i = 1 : Nr
        for j = 1 : Nr
            if abs(i - j) < 30
                C_R(i, j) = rho^(abs(i - j));
            end
        end
    end
    C_R = sqrtm(C_R);
    C_T = eye(Ns, Ns);
    for i = 1 : Ns
        for j = 1 : Ns
            if abs(i - j) < 30
                C_T(i, j) = rho^(abs(i - j));
            end
        end
    end
    C_T = sqrtm(C_T);


    G = sqrt( 1/2 ) * (randn(Nr,Ns) + randn(Nr, Ns)*1i);
    A = C_R * G * C_T;
%     H = A/norm(A,'fro')*sqrt(N);  % Normalization
end


