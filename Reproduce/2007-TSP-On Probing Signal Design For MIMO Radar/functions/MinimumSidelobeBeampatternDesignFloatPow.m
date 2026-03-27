


function  R = MinimumSidelobeBeampatternDesignFloatPow(c, M, theta0, theta1, theta2, Omega)
    fprintf('\n求解阵元功率浮动版本...\n');
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

    
    cvx_begin sdp quiet
        variable R(M,M) hermitian
        variable t
        maximize t
        subject to
            % 主瓣 - 旁瓣差约束
            for k = 1:N_Omega
                real(a0'*R*a0 - a_Omega(:,k)'*R*a_Omega(:,k)) >= t;
            end
            % 3dB 宽度约束
            real(a1'*R*a1) == 0.5 * real(a0'*R*a0);
            real(a2'*R*a2) == 0.5 * real(a0'*R*a0);
            % 半正定
            R >= 0;
            % 总功率约束
            sum(diag(R)) == c;
            % 阵元功率上下界
            for m = 1:M
                R(m,m) >= 0.8 * c/M;
                R(m,m) <= 1.2 * c/M;
            end
    cvx_end
    
    if strcmp(cvx_status, 'Solved')
        fprintf('浮动功率版本求解成功，最优 t = %f\n', t);
        R_float = R;
    else
        error('浮动功率版本求解失败');
    end

end