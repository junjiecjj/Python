



function [R, b] = helperMMSECovariance_direct(normalizedPos, Pdesired, ang, Pt)
    % 输入：
    %   elPos    - 1×N ULA 阵元位置，单位 wavelength
    %   Pdesired - 期望方向图，在角度网格 ang 上给定
    %   ang      - 角度网格，单位 degree
    %   Pt       - 总发射功率
    % 输出：
    %   R        - N×N 发射协方差矩阵
    %   b        - 期望方向图缩放因子
    if nargin < 4
        Pt = 1;
    end
    N = length(normalizedPos);
    ang = ang(:).';
    Pdesired = Pdesired(:).';
    A = steeringMatrixULA1D(normalizedPos, ang);
    L = length(ang);
    cvx_begin quiet sdp
        variable R(N, N) hermitian
        variable b nonnegative
        expression u(L)
        for l = 1:L
            a_l = A(:, l);
            u(l) = b * Pdesired(l) - real(a_l' * R * a_l);
        end
        minimize norm(u, 2)
        subject to
            R == hermitian_semidefinite(N);
            trace(R) == Pt;
    cvx_end
end