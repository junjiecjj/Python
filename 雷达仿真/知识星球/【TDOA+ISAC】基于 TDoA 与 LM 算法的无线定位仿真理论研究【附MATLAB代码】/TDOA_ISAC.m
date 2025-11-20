

clear; close all; clc;
%% ======================= 用户区参数（可改） ==========================
rng(42);                          % 固定随机种子
c = 299792458;                   % 光速 (m/s)

areaSize = [60, 40];             % 场地大小 [X宽, Y高] (m)
TRP = [ 0,          0;           % 4 个基站（可改/可增减）
        areaSize(1),0;
        areaSize(1),areaSize(2);
        0,          areaSize(2) ];
refIdx = 1;                      % 参考基站索引（做差的基站）

N_UE = 200;                      % 本次仿真的 UE 数量
SNRdB = 25;                      % 接收信噪比 (dB)
enableNLoS = true;               % 是否启用 NLoS
useLoSOnly = true;               % TDoA 解算时仅用 LoS 基站（需≥3个LoS+参考）

losProb = 0.65;                  % 每条链路为 LoS 的概率（仅用于仿真标签）
reflAtten = 0.6;                 % NLoS 反射额外衰减系数（幅度）
fs = 61.44e6;                    % 采样率（越高→TOA分辨率越高）
zcLen = 1023;                    % Zadoff-Chu 长度（探针序列）
guardZeros = 2048;               % 序列前后补零，避免卷积越界

% 环境散射体
Nscatter = 20;                   % 散射体数量
showOneUEPaths = true;           % 额外绘出一个 UE 的 NLoS 路径
uShow = 1;                       % 显示第几个 UE 的路径（若存在 NLoS）

% LM（Levenberg–Marquardt）参数（稳健求解 TDoA）
maxIter = 50; 
tol = 1e-6; 
lambda0 = 1e-2; 
lambdaGrow = 10; 
lambdaShrink = 0.3; 
lambdaMax = 1e9;
wFloor = 0.05; 
stepClip = 5.0;

% 可视化选项
plotGeometry = true;             % 几何图（含散射体、真值/估计/误差向量）
plotCDF = true;                  % 误差 CDF
plotHist = true;                 % 误差直方图
plotHeat = true;                 % 区域误差热力图（中位数）
showVectors = true; 
vecMaxShow = 400;
heatBins = [12, 8];

%% ======================= 生成散射体 ================================
SC = [ areaSize(1)*rand(Nscatter,1), areaSize(2)*rand(Nscatter,1) ]; % Nscatter×2
SC_used_count = zeros(Nscatter,1);        % 记录被选中次数（全部 UE 的所有 NLoS 链路）

%% ======================= 预先生成“探针序列” =========================
root = 29; Nzc = zcLen;
n = (0:Nzc-1).';
zc = exp(-1j*pi*root*n.*(n+1)/Nzc);      % ZC 基带复序列（CAZAC）
tx = [zeros(guardZeros,1); zc; zeros(guardZeros,1)];
Nt = numel(tx);
tgrid = (0:Nt-1).'/fs;

M = size(TRP,1);
TRP = double(TRP);
SNRlin = 10.^(SNRdB/10);

%% ======================= 结果缓存 ==========================
UE_true = zeros(N_UE,2);
UE_est  = nan(N_UE,2);
posErr  = nan(N_UE,1);
iters   = zeros(N_UE,1);
usedK   = zeros(N_UE,1);
fallbackCnt = 0; failSolve = 0;

% 记录每个 UE / 基站使用了哪个散射体（0 表示 LoS）
usedSIdx = zeros(N_UE, M);

%% ======================= 主循环：逐 UE 仿真 ==========================
for u = 1:N_UE
    % ---------- 随机 UE 位置 ----------
    UE = [2 + (areaSize(1)-4)*rand, 2 + (areaSize(2)-4)*rand];
    UE_true(u,:) = UE;

    % ---------- LoS/NLoS 标签 ----------
    isLoS = rand(M,1) < losProb;
    isLoS(refIdx) = true;

    % ---------- 直达距离 ----------
    dist_LOS = sqrt(sum((TRP - UE).^2,2));      % M×1

    % ---------- 生成接收信号（LoS：直达；NLoS：经单次散射） ----------
    rx = zeros(Nt,M);
    for m = 1:M
        if enableNLoS && ~isLoS(m)
            % 计算所有散射体的单跳路径长度：|UE-S| + |S-TRP_m|
            dxUE = SC(:,1) - UE(1);
            dyUE = SC(:,2) - UE(2);
            rUE  = sqrt(dxUE.^2 + dyUE.^2);

            dxTR = SC(:,1) - TRP(m,1);
            dyTR = SC(:,2) - TRP(m,2);
            rTR  = sqrt(dxTR.^2 + dyTR.^2);

            totalL = rUE + rTR;                               % Nscatter×1
            [d_bounce, idxS] = min(totalL);
            if d_bounce <= dist_LOS(m) + 1e-6                 % 三角不等式的等号保护
                d_bounce = dist_LOS(m) + 1e-6;
            end
            tau_m = d_bounce / c;
            usedSIdx(u,m) = idxS;                             % 记录所用散射体
            SC_used_count(idxS) = SC_used_count(idxS) + 1;

            % 振幅：路径损耗 ~ 1/d，附加反射衰减
            amp = reflAtten / max(d_bounce,1);
        else
            tau_m = dist_LOS(m) / c;                          % LoS
            amp = 1 / max(dist_LOS(m),1);
            usedSIdx(u,m) = 0;
        end

        % 生成分数时延的接收波形 + 加噪
        tm  = tgrid - tau_m;
        sm  = interp1(tgrid, tx, tm, 'linear', 0);
        % 以 amp 缩放，同时统一噪声方差基准（按 amp^2 近似）
        sigPow = mean(abs(tx).^2) * amp^2;
        noiseVar = sigPow./SNRlin;
        rx(:,m) = amp*sm + sqrt(noiseVar/2).*(randn(Nt,1)+1j*randn(Nt,1));
    end

    % ---------- 匹配滤波 + 三点插值估 TOA ----------
    est_tau = zeros(M,1);
    peakAmp = zeros(M,1);
    for m = 1:M
        [cc,lags] = xcorr(rx(:,m), tx);
        abscc = abs(cc);
        [~,I] = max(abscc);
        if I>1 && I<numel(cc)
            y1=abscc(I-1); y2=abscc(I); y3=abscc(I+1);
            denom = (y1 - 2*y2 + y3);
            if abs(denom) > eps
                delta = 0.5*(y1 - y3)/denom;  % 分数偏移（-1..+1）
            else
                delta = 0;
            end
        else
            delta = 0;
        end
        lagRefined = lags(I) + delta;
        est_tau(m) = lagRefined / fs;
        peakAmp(m) = max(abscc(I),eps);
    end

    % ---------- 选择用于 TDoA 的基站 ----------
    useIdx = (1:M).';
    if useLoSOnly
        useIdx = find(isLoS);
        if ~ismember(refIdx, useIdx)
            useIdx = unique([refIdx; useIdx(:)]);
        end
    end
    if numel(useIdx) < 3
        useIdx = (1:M).';
        fallbackCnt = fallbackCnt + 1;
    end
    K = numel(useIdx); usedK(u) = K;

    refLocal = find(useIdx==refIdx,1);
    if isempty(refLocal)
        useIdx = unique([refIdx; useIdx(:)]);
        refLocal = 1; K = numel(useIdx); usedK(u)=K;
    end

    tauSel  = est_tau(useIdx);
    TRPSel  = TRP(useIdx,:);
    peakSel = peakAmp(useIdx);

    % ---------- 相对延时与差距程 ----------
    dTau = tauSel - tauSel(refLocal);            % K×1
    meas = c * dTau(2:end);                      % (K-1)×1

    % ---------- 简单量测门控 ----------
    diagArea = hypot(areaSize(1), areaSize(2));
    % 基站间最大间距（无 pdist）
    maxPair = 0;
    for i=1:K-1
        dx = TRPSel(i,1) - TRPSel(i+1:K,1);
        dy = TRPSel(i,2) - TRPSel(i+1:K,2);
        v = sqrt(dx.^2 + dy.^2);
        if ~isempty(v), maxPair = max(maxPair, max(v)); end
    end
    geoBound = diagArea + maxPair;
    if any(~isfinite(meas)) || any(abs(meas) > 2*geoBound)
        failSolve = failSolve + 1; continue;
    end

    % ---------- LM 求解（矢量化） ----------
    p = mean(TRPSel,1).'; lam = lambda0; ok = false;
    idxNR = 1:K; idxNR(refLocal) = [];
    w = peakSel; w = w / max(w + (max(w)==0)); w = max(w, wFloor);
    wnr = w(idxNR); sqrtw = sqrt(wnr);

    % 初值残差/J
    s1 = TRPSel(refLocal,:).';
    dx1 = p(1)-s1(1); dy1 = p(2)-s1(2); d1 = max(sqrt(dx1^2+dy1^2), 1e-9);
    sk  = TRPSel(idxNR,:).';
    dxk = p(1)-sk(1,:); dyk = p(2)-sk(2,:); dk = sqrt(dxk.^2 + dyk.^2); dk = max(dk,1e-9);
    r   = (dk - d1).' - meas;
    J   = [ (dxk./dk).' - dx1/d1, (dyk./dk).' - dy1/d1 ];
    Jw  = J .* sqrtw; rw = r .* sqrtw; cost0 = rw.'*rw;

    for it = 1:maxIter
        H = Jw.'*Jw; g = Jw.'*rw;
        if ~isfinite(rcond(H)) || rcond(H) < 1e-12, lam = min(lam*lambdaGrow, lambdaMax); end
        Hd = H + lam*diag(max(diag(H),1));
        if ~isfinite(rcond(Hd)) || rcond(Hd) < 1e-15 || any(~isfinite(g))
            lam = min(lam*lambdaGrow, lambdaMax);
            if lam>=lambdaMax, break; end
            continue;
        end
        dp = - Hd \ g;
        ndp = norm(dp); if ndp > stepClip, dp = dp * (stepClip/ndp); end
        p_new = p + dp;

        % 新 r/J/cost
        dx1 = p_new(1)-s1(1); dy1 = p_new(2)-s1(2); d1 = max(sqrt(dx1^2+dy1^2), 1e-9);
        dxk = p_new(1)-sk(1,:); dyk = p_new(2)-sk(2,:); dk = sqrt(dxk.^2 + dyk.^2); dk = max(dk,1e-9);
        r_new = (dk - d1).' - meas; J_new = [ (dxk./dk).' - dx1/d1, (dyk./dk).' - dy1/d1 ];
        rw_new = r_new .* sqrtw; cost_new = rw_new.'*rw_new;

        if isfinite(cost_new) && cost_new < cost0
            p = p_new; r = r_new; J = J_new; Jw = J .* sqrtw; rw = rw_new; cost0 = cost_new;
            lam = max(lam*lambdaShrink, 1e-12);
            if norm(dp) < tol, ok = true; break; end
        else
            lam = min(lam*lambdaGrow, lambdaMax);
            if lam >= lambdaMax, break; end
        end
    end

    if ~ok || any(~isfinite(p))
        failSolve = failSolve + 1; continue;
    end

    UE_est(u,:) = p(:).';
    posErr(u)   = norm(UE - UE_est(u,:));
    iters(u)    = it;
end

%% ======================= 统计结果（忽略 NaN） =======================
e = posErr; good = isfinite(e); Nvalid = sum(good);
if Nvalid==0, error('全部解算失败；请检查几何/参数。'); end
es = sort(e(good)); cdfy = (1:numel(es)).'/numel(es);
P10 = es( max(1, ceil(0.10*numel(es))) ); P50 = es( max(1, ceil(0.50*numel(es))) ); P90 = es( max(1, ceil(0.90*numel(es))) );

fprintf('=== TDoA 多UE统计（含散射体的NLoS模型） ===\n');
fprintf('UE 总数: %d, 成功: %d, 失败: %d（LoSOnly兜底: %d 次）\n', N_UE, Nvalid, failSolve, fallbackCnt);
fprintf('误差:  均值=%.3f m, 中位=%.3f m, P10=%.3f m, P90=%.3f m, 最大=%.3f m\n', mean(e(good)), P50, P10, P90, max(e(good)));
fprintf('迭代次数: 均值=%.2f, 中位=%.0f（成功样本）\n', mean(iters(good)), median(iters(good)));
fprintf('每次解算参与基站数（均值，成功样本）: %.2f\n', mean(usedK(good)));

%% ======================= 可视化 1：几何图（含散射体） =================
if plotGeometry
    figure('Name','Geometry with Scatterers'); clf
    hold on; axis equal; grid on;
    rectangle('Position',[0,0,areaSize],'EdgeColor',[0.6 0.6 0.6]);
    % 散射体使用热度：用颜色表示次数
    sCount = SC_used_count;
    sCountScaled = sCount;                     % 颜色 = 使用次数
    hSC = scatter(SC(:,1), SC(:,2), 20 + 3*sCount, sCountScaled, 'filled'); % 点越用越大
    colorbar; ylabel(colorbar, '散射体被使用次数');
    hTRP  = plot(TRP(:,1), TRP(:,2), 'ks', 'MarkerSize',8, 'LineWidth',1.5);
    hTrue = plot(UE_true(good,1), UE_true(good,2), 'g.', 'MarkerSize',12);
    hEst  = plot(UE_est(good,1),  UE_est(good,2),  'rx', 'MarkerSize',6,  'LineWidth',1);

    if showVectors
        idxv = find(good);
        nShow = min(vecMaxShow, numel(idxv));
        idxv = idxv(randperm(numel(idxv), nShow));
        quiver(UE_true(idxv,1), UE_true(idxv,2), UE_est(idxv,1)-UE_true(idxv,1), UE_est(idxv,2)-UE_true(idxv,2), 0, 'Color',[0.85 0.2 0.2], 'LineWidth',0.8);
    end
    legend([hTRP(1), hSC(1), hTrue(1), hEst(1)], {'TRP','散射体','UE真值(成功)','UE估计'}, 'Location','bestoutside');
    title(sprintf('TDoA 多UE（含散射体）：N=%d(成功 %d), SNR=%.0f dB, fs=%.2f MHz', N_UE, Nvalid, SNRdB, fs/1e6));
    xlabel('X (m)'); ylabel('Y (m)');
end

%% ======================= 可视化 1b：单UE的NLoS路径（可选） ===========
if showOneUEPaths
    u0 = min(max(1,uShow), N_UE);
    figure('Name','One UE NLoS Paths'); clf
    hold on; axis equal; grid on;
    rectangle('Position',[0,0,areaSize],'EdgeColor',[0.7 0.7 0.7]);
    plot(TRP(:,1), TRP(:,2), 'ks', 'MarkerSize',8, 'LineWidth',1.5);
    plot(UE_true(u0,1), UE_true(u0,2), 'go', 'MarkerSize',8, 'LineWidth',1.5);
    plot(UE_est(u0,1),  UE_est(u0,2),  'rx', 'MarkerSize',8, 'LineWidth',1.5);
    scatter(SC(:,1), SC(:,2), 12, [0.6 0.6 0.6], 'filled');

    % 把该 UE 的 NLoS 链路画出来（UE->S->TRP）
    for m = 1:M
        k = usedSIdx(u0,m);
        if k>0
            S = SC(k,:);
            plot([UE_true(u0,1), S(1)], [UE_true(u0,2), S(2)], 'm--', 'LineWidth',1.2);
            plot([S(1), TRP(m,1)], [S(2), TRP(m,2)], 'm--', 'LineWidth',1.2);
            plot(S(1), S(2), 'md', 'MarkerSize',6, 'LineWidth',1.2); % 高亮该散射体
        end
    end
    legend('TRP','UE真值','UE估计','散射体(灰)','NLoS路径','Location','bestoutside');
    title(sprintf('单UE的NLoS路径示意（UE #%d）', u0));
    xlabel('X (m)'); ylabel('Y (m)');
end

%% ======================= 可视化 2：误差 CDF ==========================
if plotCDF
    figure('Name','Error CDF'); clf; grid on; hold on;
    plot(es, cdfy, 'LineWidth',1.5);
    yl = ylim;
    plot([P10 P10], yl, 'k--'); text(P10, 0.05, sprintf('P10=%.2f m',P10), 'HorizontalAlignment','left','VerticalAlignment','bottom');
    plot([P50 P50], yl, 'k--'); text(P50, 0.35, sprintf('P50=%.2f m',P50), 'HorizontalAlignment','left','VerticalAlignment','bottom');
    plot([P90 P90], yl, 'k--'); text(P90, 0.75, sprintf('P90=%.2f m',P90), 'HorizontalAlignment','left','VerticalAlignment','bottom');
    xlabel('定位误差 (m)'); ylabel('概率'); title('误差 CDF（成功样本）');
end

%% ======================= 可视化 3：误差直方图 =======================
if plotHist
    figure('Name','Error Histogram'); clf; grid on;
    histogram(e(good), max(10,round(sqrt(Nvalid))), 'Normalization','pdf');
    xlabel('定位误差 (m)'); ylabel('概率密度'); title('误差直方图（成功样本）');
end

%% ======================= 可视化 4：区域误差热力图（中位数） =========
if plotHeat
    nx = heatBins(1); ny = heatBins(2);
    xedges = linspace(0, areaSize(1), nx+1);
    yedges = linspace(0, areaSize(2), ny+1);
    xi = discretize(UE_true(good,1), xedges);
    yi = discretize(UE_true(good,2), yedges);
    ind = sub2ind([nx ny], xi, yi);
    medMap = accumarray(ind, e(good), [nx*ny 1], @(x) median(x,'omitnan'), NaN);
    medMap = reshape(medMap, [nx ny]);           % X×Y
    xcent = 0.5*(xedges(1:end-1)+xedges(2:end));
    ycent = 0.5*(yedges(1:end-1)+yedges(2:end));

    figure('Name','Median Error Heatmap'); clf
    imagesc(xcent, ycent, medMap.'); axis xy equal tight;
    colorbar; title('各网格中位误差 (m)（成功样本）'); xlabel('X (m)'); ylabel('Y (m)');
end
