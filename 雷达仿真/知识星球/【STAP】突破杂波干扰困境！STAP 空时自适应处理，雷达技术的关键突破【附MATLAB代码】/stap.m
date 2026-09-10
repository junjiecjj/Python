%% STAP_demo.m
% 空时自适应处理（STAP）标准理论仿真
%
% 本程序采用如下空时数据堆叠方式：
%   x = [x_1^T, x_2^T, ..., x_N^T]^T
% 其中 x_n 为第 n 个阵元在 M 个脉冲上的慢时间数据。
% 因此空时导向矢量统一写为：
%   s(theta,nu) = a(theta) \otimes b(nu)
%
% 本程序重点演示：
%   1) 空间导向矢量与慢时间多普勒导向矢量；
%   2) 旁视阵列条件下的杂波脊；
%   3) 杂波、宽带压制式干扰机和白噪声的协方差建模；
%   4) 最大输出 SINR / MVDR 形式的 STAP 最优权重；
%   5) 图1：四种干扰场景下的 STAP 角度-多普勒二维响应；
%   6) 图2：均匀空时匹配与 Chebyshev 空时加窗的二维响应对比。
%
% 说明：
%   这里直接使用“理论协方差矩阵”构造 STAP 权重，用于清楚展示
%   STAP 的基本原理。实际系统中通常需由邻近训练距离单元估计协方差矩阵。

clear;
clc;
close all;

%% 1. 基本参数

N = 16;                         % 阵元数
M = 16;                         % 相干处理脉冲数
NM = N * M;                     % 空时维数

lambda = 0.03;                  % 工作波长 / m
d = lambda / 2;                 % 阵元间距 / m
v = 60;                         % 平台速度 / (m/s)
Tr = 1e-4;                      % 脉冲重复周期 PRI / s

% 归一化杂波脊斜率：
% nu_c = beta * u
% 其中 u = (d/lambda) sin(theta)，nu = f_d Tr
beta = 2 * v * Tr / d;

% 目标参数
thetaTargetDeg = 10;            % 目标角度 / deg
nuTarget = 0.32;                % 目标归一化多普勒频率 / cycles per PRI

% 噪声、杂波和干扰功率参数
noisePower = 1;                 % 每个空时通道的白噪声功率
CNRdB = 40;                     % 总杂波噪声比 / dB
JNRdB = 35;                     % 每个干扰机的干扰噪声比 / dB

% 杂波离散角度
numClutterPatches = 181;
clutterAngleDeg = linspace(-90, 90, numClutterPatches);

% 两个宽带压制式干扰机，仅具有空间方向性，慢时间近似白
jammerAngleDeg = [-35, 40];

% 二维响应扫描网格
angleScanDeg = -90:1:90;
nuScan = -0.5:0.005:0.5;

%% 2. 目标空时导向矢量

aTarget = spatialSteeringVector(N, d, lambda, thetaTargetDeg);
bTarget = dopplerSteeringVector(M, nuTarget);

% 按照当前堆叠方式，空时导向矢量为 a \otimes b
sTarget = kron(aTarget, bTarget);

%% 3. 白噪声协方差矩阵

Rn = noisePower * eye(NM);

%% 4. 杂波协方差矩阵

% 旁视阵列、静止地面杂波条件下：
%   u(theta) = (d/lambda) sin(theta)
%   nu_c(theta) = beta * u(theta)
clutterSpatialFreq = (d / lambda) * sind(clutterAngleDeg);
clutterDoppler = beta * clutterSpatialFreq;

% 将总杂波功率均匀分配到所有独立杂波块
totalClutterPower = noisePower * 10^(CNRdB / 10);
clutterPatchPower = totalClutterPower / numClutterPatches;

Vclutter = zeros(NM, numClutterPatches);

for k = 1:numClutterPatches
    aClutter = spatialSteeringVector( ...
        N, d, lambda, clutterAngleDeg(k));

    bClutter = dopplerSteeringVector( ...
        M, clutterDoppler(k));

    Vclutter(:, k) = kron(aClutter, bClutter);
end

% 各杂波块相互不相关且等功率：
%   Rc = sum_k sigma_c,k^2 s_c,k s_c,k^H
%      = sigma_c^2 Vc Vc^H
Rc = clutterPatchPower * (Vclutter * Vclutter');

% 数值上强制 Hermitian，对理论结果无影响
Rc = (Rc + Rc') / 2;

%% 5. 宽带压制式干扰机协方差矩阵

% 对于第 q 个宽带压制式干扰机：
%   Rj,q = Pj (a_j a_j^H) \otimes I_M
%
% 该模型表示：
%   - 空间上来自固定到达角；
%   - 慢时间上近似白，因此会占据该角度处的一整条多普勒方向。

Rj = zeros(NM);

jammerPower = noisePower * 10^(JNRdB / 10);

for q = 1:numel(jammerAngleDeg)
    aJammer = spatialSteeringVector( ...
        N, d, lambda, jammerAngleDeg(q));

    Rj = Rj + jammerPower * ...
        kron(aJammer * aJammer', eye(M));
end

Rj = (Rj + Rj') / 2;

%% 6. 四种场景下的干扰加噪声协方差矩阵

Rcase = cell(4, 1);
caseName = cell(4, 1);

Rcase{1} = Rn;
caseName{1} = '仅白噪声';

Rcase{2} = Rn + Rc;
caseName{2} = '白噪声 + 杂波';

Rcase{3} = Rn + Rj;
caseName{3} = '白噪声 + 干扰机';

Rcase{4} = Rn + Rc + Rj;
caseName{4} = '白噪声 + 杂波 + 干扰机';

%% 7. 计算 STAP 最优权重与二维角度-多普勒响应

Wstap = cell(4, 1);
responseDB = cell(4, 1);

for iCase = 1:4

    R = Rcase{iCase};

    % 最大输出 SINR 的权重方向为：
    %   w \propto R^{-1} s0
    %
    % 加入无失真约束 w^H s0 = 1 后，得到 MVDR/STAP 权重：
    %   w = R^{-1}s0 / (s0^H R^{-1}s0)
    %
    % 使用反斜杠求解线性方程，避免显式计算 inv(R)。
    x = R \ sTarget;
    Wstap{iCase} = x / (sTarget' * x);

    response = computeSpaceTimeResponse( ...
        Wstap{iCase}, N, M, d, lambda, ...
        angleScanDeg, nuScan);

    % 因为采用 w^H s0 = 1 的归一化，理论目标响应为 0 dB。
    responseDB{iCase} = 20 * log10(max(response, eps));
    responseDB{iCase}(responseDB{iCase} < -80) = -80;
end

%% 8. 绘制四种场景的角度-多普勒响应

figure('Color', 'w');

for iCase = 1:4

    subplot(2, 2, iCase);

    imagesc(angleScanDeg, nuScan, responseDB{iCase});
    axis xy;
    axis tight;

    xlabel('角度 \theta / deg');
    ylabel('归一化多普勒 \nu = f_d T_r');
    title(caseName{iCase});
    colorbar;
    caxis([-80, 0]);

    hold on;

    % 标出目标位置
    plot(thetaTargetDeg, nuTarget, 'wo', ...
        'MarkerSize', 7, 'LineWidth', 1.5);

    % 对含杂波的场景叠加理论杂波脊
    if iCase == 2 || iCase == 4
        nuRidge = beta * (d / lambda) * sind(angleScanDeg);

        validIndex = abs(nuRidge) <= 0.5;

        plot(angleScanDeg(validIndex), ...
             nuRidge(validIndex), ...
             'w--', 'LineWidth', 1.0);
    end

    % 对含干扰机的场景标出干扰机到达角
    if iCase == 3 || iCase == 4
        for q = 1:numel(jammerAngleDeg)
            xline(jammerAngleDeg(q), 'w:', ...
                'LineWidth', 1.0);
        end
    end

    hold off;
end

sgtitle('图1  STAP 在不同干扰环境下的角度-多普勒二维响应');

%% 9. 图2：均匀空时匹配与 Chebyshev 空时加窗对比

% 这一部分不是 STAP 的协方差自适应权重设计，而是固定空时加窗。
% 其目的与经典示例中的第二幅图一致：比较不加窗与 Chebyshev 加窗
% 后的角度-多普勒二维旁瓣特性。
%
% 未加窗空时匹配权重：
%   w_uniform = s0 / (s0^H s0)
%
% 空时 Chebyshev 窗：
%   g = g_space \otimes g_time
%
% 加窗后：
%   w_cheb ∝ s0 .* g
%
% 两种权重都归一化到目标方向单位增益，便于公平比较。

% 未加窗的空时匹配权重
wUniform = sTarget / (sTarget' * sTarget);

% 分别在空间维和慢时间维构造 Chebyshev 窗
spaceWindow = chebwin(N, 30);
timeWindow = chebwin(M, 30);

% 联合空时窗，与 s = a \otimes b 的堆叠方式保持一致
spaceTimeWindow = kron(spaceWindow, timeWindow);

% 将固定空时窗作用在目标空时导向矢量上
wChebRaw = sTarget .* spaceTimeWindow;

% 归一化，使目标位置满足 w^H s0 = 1
wCheb = wChebRaw / (sTarget' * wChebRaw);

% 计算未加窗和加窗后的二维响应
responseUniform = computeSpaceTimeResponse( ...
    wUniform, N, M, d, lambda, angleScanDeg, nuScan);

responseCheb = computeSpaceTimeResponse( ...
    wCheb, N, M, d, lambda, angleScanDeg, nuScan);

responseUniformDB = 20 * log10(max(responseUniform, eps));
responseChebDB = 20 * log10(max(responseCheb, eps));

responseUniformDB(responseUniformDB < -80) = -80;
responseChebDB(responseChebDB < -80) = -80;

figure('Color', 'w');

subplot(1, 2, 1);
imagesc(angleScanDeg, nuScan, responseUniformDB);
axis xy;
axis tight;
xlabel('角度 \theta / deg');
ylabel('归一化多普勒 \nu = f_d T_r');
title('均匀空时匹配');
colorbar;
caxis([-80, 0]);

subplot(1, 2, 2);
imagesc(angleScanDeg, nuScan, responseChebDB);
axis xy;
axis tight;
xlabel('角度 \theta / deg');
ylabel('归一化多普勒 \nu = f_d T_r');
title('Chebyshev 空时加窗');
colorbar;
caxis([-80, 0]);

sgtitle('图2  空时 Chebyshev 加窗前后的二维响应对比');

%% 10. 输出 SINR 对比

% 组合场景：白噪声 + 杂波 + 干扰机
Rin = Rn + Rc + Rj;

% 常规空时匹配权重，只对准目标，不利用干扰协方差
wMatched = sTarget / (sTarget' * sTarget);

% STAP 最优权重
wSTAP = Wstap{4};

% 设目标功率为 1，则输出 SINR 为：
%   SINR_out = |w^H s0|^2 / (w^H Rin w)
sinrMatched = abs(wMatched' * sTarget)^2 / ...
    real(wMatched' * Rin * wMatched);

sinrSTAP = abs(wSTAP' * sTarget)^2 / ...
    real(wSTAP' * Rin * wSTAP);

fprintf('--------------------------------------------------\n');
fprintf('组合场景输出 SINR 对比\n');
fprintf('常规空时匹配权重： %.2f dB\n', 10 * log10(sinrMatched));
fprintf('STAP 最优权重：     %.2f dB\n', 10 * log10(sinrSTAP));
fprintf('STAP SINR 增益：    %.2f dB\n', ...
    10 * log10(sinrSTAP / sinrMatched));
fprintf('--------------------------------------------------\n');

%% 局部函数

function a = spatialSteeringVector(N, d, lambda, thetaDeg)
%SPATIALSTEERINGVECTOR 生成 ULA 空间导向矢量
%
%   a(theta) =
%   [1, exp(j2piu), ..., exp(j2pi(N-1)u)]^T
%
%   其中：
%       u = (d/lambda) sin(theta)

    elementIndex = (0:N-1).';
    u = (d / lambda) * sind(thetaDeg);

    a = exp(1j * 2 * pi * elementIndex * u);
end


function b = dopplerSteeringVector(M, nu)
%DOPPLERSTEERINGVECTOR 生成慢时间多普勒导向矢量
%
%   b(nu) =
%   [1, exp(j2pinu), ..., exp(j2pi(M-1)nu)]^T
%
%   nu = f_d Tr 为归一化多普勒频率，单位为 cycles per PRI。

    pulseIndex = (0:M-1).';

    b = exp(1j * 2 * pi * pulseIndex * nu);
end


function response = computeSpaceTimeResponse( ...
    w, N, M, d, lambda, angleScanDeg, nuScan)
%COMPUTESPACETIMERESPONSE 计算角度-多普勒二维空时响应
%
%   response(iNu,iTheta)
%       = |w^H s(theta_i,nu_j)|
%
%   其中：
%       s(theta,nu) = a(theta) \otimes b(nu)

    numAngle = numel(angleScanDeg);
    numDoppler = numel(nuScan);

    response = zeros(numDoppler, numAngle);

    % 预先构造所有慢时间多普勒导向矢量
    pulseIndex = (0:M-1).';
    Bscan = exp(1j * 2 * pi * pulseIndex * nuScan);

    for iAngle = 1:numAngle

        aScan = spatialSteeringVector( ...
            N, d, lambda, angleScanDeg(iAngle));

        % Sscan 的每一列分别为：
        %   a(theta_i) \otimes b(nu_j)
        Sscan = kron(aScan, Bscan);

        response(:, iAngle) = abs(w' * Sscan).';
    end
end
