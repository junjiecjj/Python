

function [GLRT, peaks, est_theta] = GLRTBeampatterns(M, Rxx, Ryx, Ryy, theta_grid)
    
    L = length(theta_grid);
    % 导向矢量函数 (均匀线阵，半波长间距)
    afun = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));
    GLRT = zeros(L, 1);

    %% 扫描每个角度
    for idx = 1:L
        theta = theta_grid(idx);
        at = afun(theta);       % 发射导向矢量 M×1
        ar = afun(theta);     % 接收导向矢量 M×1
        Q = Ryy - (Ryx * conj(at) * at.' * Ryx') / (at.' * Rxx * conj(at));
        % 分母项: a^* Q^{-1} a
        denom_GLRT = ar' / Q * ar;
        GLRT(idx) = 1 - ar' / Ryy * ar / denom_GLRT;
    end
    GLRT = abs(GLRT);
    GLRT_norm = GLRT / max(GLRT);
    [~, locs] = findpeaks(GLRT_norm, 'MinPeakHeight', 0.6*max(GLRT_norm), 'MinPeakDistance', 5);
    peaks = GLRT(locs);
    est_theta = theta_grid(locs);
end