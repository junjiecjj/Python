function H=relatedchannel(M,N,rho)
% rho = 0.2;   % 相关系数[0,1）越大越相关
    C_R = eye(M*N, M*N);
    for i = 1 : M*N
        for j = 1 : M*N
            if abs(i - j) < 30
                C_R(i, j) = rho^(abs(i - j));
            end
        end
    end
    C_R = sqrtm(C_R);
    C_T = eye(M*N, M*N);
    for i = 1 : M*N
        for j = 1 : M*N
            if abs(i - j) < 30
                C_T(i, j) = rho^(abs(i - j));
            end
        end
    end
    C_T = sqrtm(C_T);


    G = sqrt( 1/2 ) * (randn(M*N, M*N) + randn(M*N, M*N)*1i);
    A = C_R * G * C_T;
    H = A/norm(A,'fro')*sqrt(N);  % Normalization
end


