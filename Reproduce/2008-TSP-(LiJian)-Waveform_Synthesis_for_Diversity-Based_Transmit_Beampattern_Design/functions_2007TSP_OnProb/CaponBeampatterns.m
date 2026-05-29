


function [Capon, peaks, est_theta] = CaponBeampatterns(M, Rxx, Ryx, Ryy, theta_grid)
    L = length(theta_grid);
    % 导向矢量函数 (均匀线阵，半波长间距)
    afun = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));
    Capon = zeros(L, 1);
    %% 扫描每个角度
    % ========== Capon 空间谱 严格按照公式(36) ==========
    for idx = 1:L
        theta = theta_grid(idx);
        at = afun(theta);
        ar = afun(theta);
        % 分子: a^* Ryy^{-1} Ryx ac
        num_capon = ar' / Ryy * Ryx * conj(at);   % 标量
        % 分母: a^* Ryy^{-1} a  *  a^T Rxx ac
        denom_capon = (ar' / Ryy * ar) * (at.' * Rxx * conj(at));
        % Capon 谱值
        Capon(idx) = (num_capon) / (denom_capon);   % 通常取模平方，保证为正
    end
    Capon = abs(Capon);
    Capon_norm = Capon / max(Capon);
    [~, locs] = findpeaks(Capon_norm, 'MinPeakHeight', 0.6*max(Capon_norm), 'MinPeakDistance', 5);
    peaks = Capon(locs);
    est_theta = theta_grid(locs);
end