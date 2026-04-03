


function  R = MinimumSidelobeBeampatternDesignFloatSidelobe(c, M, theta0, theta1, theta2, Omega)
    fprintf('\n求解旁瓣浮动版本...\n');
    N_Omega = length(Omega);

    % 导向矢量函数（半波长间距）
    a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));
    
    % 预计算导向矢量
    a0 = a(theta0);
    a1 = a(theta1);
    a2 = a(theta2);
    a_Omega = zeros(M, N_Omega);
    for k = 1:N_Omega
        a_Omega(:,k) = a(Omega(k));
    end

    delta = 0.1;           % 3dB 宽度松弛量
    cvx_begin sdp quiet
        variable R(M,M) hermitian
        variable t
        maximize t
        subject to
            % 主瓣 - 旁瓣差约束：每个旁瓣点的主瓣功率差 ≥ t
            for k = 1:N_Omega
                real(a0'*R*a0 - a_Omega(:,k)'*R*a_Omega(:,k)) >= t;
            end
            % 放宽的 3dB 宽度约束（允许在 0.5±δ 范围内）
            P0 = real(a0'*R*a0);
            real(a1'*R*a1) >= (0.5 - delta) * P0;
            real(a1'*R*a1) <= (0.5 + delta) * P0;
            real(a2'*R*a2) >= (0.5 - delta) * P0;
            real(a2'*R*a2) <= (0.5 + delta) * P0;
            
            % 半正定
            R >= 0;
            
            % 阵元功率固定为 c/M
            for m = 1:M
                R(m,m) == c(m)/M;
            end
    cvx_end
    
    if strcmp(cvx_status, 'Solved')
        fprintf('求解成功，最优 t = %f\n', t);
        R_float1 = R;
    else
        error('求解失败，请检查问题可行性或调整 δ');
    end

end