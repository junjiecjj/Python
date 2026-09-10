%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 目的：               旁视阵列条件下的全维自适应空时处理（STAP）
% 描述：               原注释写为“单目标和双目标情况”，但当前代码实际包含：
%                      1 个目标、1 个干扰机、地杂波和白噪声
% 参考文献：
% [1] James Ward, "Space-Time Adaptive Processing for Airborne
%     Radar". MIT Lincoln Lab tech report 1015, 1994.
% [2] J. R. Guerci, "Space-Time Adaptive Processing for Radar".
%     Artech House, 2003.
% 版本：
% $Revision:	     $1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear;
lambda = 0.03; % 工作波长；在纯阵列信号处理的归一化模型中并非必须显式给出
d = lambda/2; % 阵元间距

N = 10; % ULA 阵元数
M = 12; % 每个 CPI 内的脉冲数

CNR = 50;   % dB，杂波噪声比
SNR = 0;    % dB，目标信噪比
JNR = 30;   % dB，干扰噪声比

noisePower = 1;
clutterPower = noisePower * 10^(CNR/10);
tgtPower = noisePower * 10^(SNR/10);
jammerPower = noisePower * 10^(JNR/10);

% 杂波脊斜率参数，对应文献 [1] 中 Eq. (3.2)
% 在归一化空间频率—归一化多普勒平面中：
%   fd_ClutterNormalized = beta * spatialFreq_normalized
beta = 1;


%% (1) Rc：杂波协方差矩阵
% -------- 地杂波脊模型，参见文献 [1] Eq. (3.2) --------
% 归一化杂波多普勒：
%   fd_ClutterNormalized = fd_Clutter / PRF
% 归一化空间频率：
%   spatialFreq_normalized = d*sin(theta)/lambda
% 旁视阵列条件下二者满足：
%   fd_ClutterNormalized = beta*spatialFreq_normalized

% （调试方式 1）直接对 sin(theta) 均匀离散
No = 250;       % 杂波块个数
sintheta = linspace(-1, 1, No);
clutterSpatialFreq_normalized = d./lambda*sintheta;

%% （调试方式 2）也可以直接对杂波方位角进行离散
%clutterAzimuth = -90:1:90; % 方位角扫描范围
%clutterSpatialFreq_normalized = d./lambda*sind(clutterAzimuth); % d/lambda = 0.5

% 根据杂波脊关系计算每个杂波块的归一化多普勒
fd_ClutterNormalized = beta*clutterSpatialFreq_normalized;

% 初始化杂波协方差矩阵
Rc = complex(zeros(M*N));

% V 的每一列保存一个杂波块的空时导向矢量
V = zeros(M*N, length(clutterSpatialFreq_normalized));

for k = 1:length(clutterSpatialFreq_normalized)

    % 第 k 个杂波块的空间导向矢量
    a_clutter = exp(-1j*2*pi*clutterSpatialFreq_normalized(k)*[0: N - 1].');

    % 第 k 个杂波块的慢时间/多普勒导向矢量
    b_clutter = exp(-1j*2*pi*fd_ClutterNormalized(k)*[0:M - 1].');

    % 空时导向矢量
    % 本代码采用 v = b(fd) \otimes a(theta)
    % 即“脉冲维在外、阵元维在内”的数据堆叠顺序
    v_clutter = sqrt(clutterPower)*kron(b_clutter, a_clutter);

    V(:, k) = v_clutter;

    % 假设各杂波块相互不相关，则总杂波协方差为各块外积协方差之和
    Rc = Rc + v_clutter*v_clutter';
end

% 对离散杂波块取平均，使总杂波功率尺度由 clutterPower 控制
Rc = Rc./length(clutterSpatialFreq_normalized);


%% (2) Rn：噪声协方差矩阵
% 假设不同阵元、不同脉冲上的接收机噪声相互不相关
Rn = noisePower*eye(M*N);


%% (3) Rt：目标协方差矩阵
tgtAzimuth = 0;

% 目标归一化空间频率
tgtSpatialFreq_normalized = d./lambda*sind(tgtAzimuth);

% 目标归一化多普勒，通常位于 [-0.5, 0.5]
fd_tgtNormalized = -0.1;

% 目标空间导向矢量
a_tgt = exp(-1j*2*pi*tgtSpatialFreq_normalized*[0: N - 1].');

% 目标慢时间/多普勒导向矢量
b_tgt = exp(-1j*2*pi*fd_tgtNormalized*[0:M - 1].');

% 目标空时导向矢量
v_tgt = sqrt(tgtPower)*kron(b_tgt, a_tgt);

% 单个确定角度—多普勒目标对应秩 1 协方差
Rt = v_tgt*v_tgt';


%% (4) Rj：干扰机协方差矩阵
jammerAzimuth = -30;

% 干扰机归一化空间频率
jammerSpatialFreq_normalized = d./lambda*sind(jammerAzimuth);

% 若希望干扰机位于杂波脊上，可使用下面一行
%fd_jammerNormalized = beta*jammerSpatialFreq_normalized;

% 当前代码人为指定干扰机归一化多普勒为 0.1
fd_jammerNormalized = 0.1;

% 干扰机空间导向矢量
a_jammer = exp(-1j*2*pi*jammerSpatialFreq_normalized*[0: N - 1].');

% 干扰机慢时间/多普勒导向矢量
b_jammer = exp(-1j*2*pi*fd_jammerNormalized*[0:M - 1].');

% 干扰机空时导向矢量
v_jammer = sqrt(jammerPower)*kron(b_jammer, a_jammer);

% 当前代码把干扰机建模为“固定角度 + 固定多普勒”的秩 1 点干扰
Rj = v_jammer*v_jammer';


%% (5) R：杂波、干扰机和噪声构成的总干扰加噪声协方差矩阵
R = Rc + Rj + Rn;

% 对 R 做奇异值分解。由于 R 为 Hermitian 半正定矩阵，
% 其奇异值与特征值一致
[U, S, V] = svd(R);

% 绘制干扰加噪声协方差矩阵的特征值谱
figure;
plot(10*log10(diag(S)));
xlabel('特征值序号');
ylabel('特征值 / dB');
title('干扰加噪声协方差矩阵特征值谱');


%% (6) 空时 DBF 功率谱与 Capon/MVDR 功率谱
% 对应文献 [1] Eq. (3.16) 与 Eq. (3.18)
% 这里将 Rt 加入 R_total，是为了显示包含目标、杂波、噪声和干扰的总回波谱
% 注意：R_total 用于谱估计；R = Rc+Rj+Rn 用于后续 STAP 权重设计

% debug:Rc1 = Rc; Rj1 = Rj; Rn1 = Rn; Rt11 = Rt; save data Rc1 Rj1 Rn1 Rt11;
R_total = Rc + Rj + Rn + Rt;

% ---> 注意(1)：作者认为直接均匀划分角度网格的效果不好
%azimuthGrid = linspace(-90, 90);%-90:1:90;
%SpatialFreqGrid_normalized = d./lambda*sind(azimuthGrid);

% <--- 注意(1)：这里改为对 sin(theta) 均匀离散，
% 使归一化空间频率网格本身均匀
sintheta = linspace(-1, 1);
SpatialFreqGrid_normalized = d./lambda*sintheta; % d/lambda = 0.5

% 构造独立的归一化多普勒扫描网格。
% beta 只决定杂波脊 fd_c = beta*f_s 的斜率，不应限制二维 Doppler 扫描范围。
fdGrid_Normalized = linspace(-0.5, 0.5, length(SpatialFreqGrid_normalized));

% 所有空间频率扫描点对应的空间导向矩阵
a_Grid = exp(-1j*2*pi*[0: N - 1].'*SpatialFreqGrid_normalized);

% 所有归一化多普勒扫描点对应的慢时间导向矩阵
b_Grid = exp(-1j*2*pi*[0:M - 1].'*fdGrid_Normalized);

% 初始化常规 DBF/Bartlett 谱
P_dbf = complex(zeros(length(fdGrid_Normalized), length(SpatialFreqGrid_normalized)));

% 初始化 Capon/MVDR 谱
P_capon = complex(zeros(length(fdGrid_Normalized), length(SpatialFreqGrid_normalized)));

for iSpatialFreq = 1:length(SpatialFreqGrid_normalized)
    for j_fdNormalized = 1:length(fdGrid_Normalized)

        % 当前二维扫描点的空时导向矢量
        v = kron(b_Grid(:, j_fdNormalized), a_Grid(:, iSpatialFreq));

        % 常规空时 DBF/Bartlett 功率谱：v^H R_total v
        P_dbf(j_fdNormalized, iSpatialFreq) = v'*R_total*v;

        % Capon/MVDR 功率谱：1/(v^H R_total^{-1}v)
        % 使用 R_total\v 代替显式求逆。
        P_capon(j_fdNormalized, iSpatialFreq) = 1./real(v'*(R_total\v));
    end
end

% 显示总回波 DBF 二维功率谱
figure;
imagesc(SpatialFreqGrid_normalized, fdGrid_Normalized, 10*log10(max(real(P_dbf), eps)))
set(gca,'ydir','normal');
colorbar;
xlabel('归一化空间频率');
ylabel('归一化多普勒频率');
title('STAP处理前总回波的DBF二维功率谱');

% 显示总回波 DBF 三维功率谱
figure;
surf(SpatialFreqGrid_normalized, fdGrid_Normalized, 10*log10(max(real(P_dbf), eps)))
shading interp;
colorbar;
xlabel('归一化空间频率');
ylabel('归一化多普勒频率');
zlabel('功率 / dB');
title('STAP处理前总回波的DBF三维功率谱');

% 显示总回波 Capon/MVDR 二维功率谱
figure;
imagesc(SpatialFreqGrid_normalized, fdGrid_Normalized, 10*log10(max(real(P_capon), eps)))
set(gca,'ydir','normal');
colorbar;
xlabel('归一化空间频率');
ylabel('归一化多普勒频率');
title('STAP处理前总回波的Capon二维功率谱');

% 显示总回波 Capon/MVDR 三维功率谱
figure;
surf(SpatialFreqGrid_normalized, fdGrid_Normalized, 10*log10(max(real(P_capon), eps)))
shading interp;
colorbar;
xlabel('归一化空间频率');
ylabel('归一化多普勒频率');
zlabel('功率 / dB');
title('STAP处理前总回波的Capon三维功率谱');


%% (7) 计算 STAP 最优权重
% 干扰加噪声协方差矩阵：R = Rc + Rj + Rn
% 最大输出 SINR 的最优权重方向：w_opt ∝ R^{-1}v_tgt。
% MATLAB 中直接求解线性方程比显式计算 inv(R) 更稳定。
wopt = R \ v_tgt;

% 若加入无失真约束 w^H v_tgt = 1，可采用归一化形式：
% x = R \ v_tgt;
% wopt = x/(v_tgt'*x);


%% (8) 最优空时滤波器的二维响应
sintheta = linspace(-1, 1);
SpatialFreqGrid_normalized = d./lambda*sintheta; % d/lambda = 0.5
% Doppler 扫描轴独立覆盖未模糊区间 [-0.5, 0.5]
fdGrid_Normalized = linspace(-0.5, 0.5, length(SpatialFreqGrid_normalized));
a_Grid = exp(-1j*2*pi*[0: N - 1].'*SpatialFreqGrid_normalized);
b_Grid = exp(-1j*2*pi*[0:M - 1].'*fdGrid_Normalized);

% 初始化最优 STAP 权重的二维响应
Y = complex(zeros(length(fdGrid_Normalized), length(SpatialFreqGrid_normalized)));

for iSpatialFreq = 1:length(SpatialFreqGrid_normalized)
    for j_fdNormalized = 1:length(fdGrid_Normalized)

        % 当前角度—多普勒扫描点的空时导向矢量
        v = kron(b_Grid(:, j_fdNormalized), a_Grid(:, iSpatialFreq));

        % 最优空时滤波器对该扫描点的响应
        Y(j_fdNormalized, iSpatialFreq) = wopt'*v;
    end
end

% 显示 STAP 最优权重的二维响应
figure;
imagesc(SpatialFreqGrid_normalized, fdGrid_Normalized, 10*log10(max(abs(Y).^2, eps)))
set(gca,'ydir','normal');
colorbar;
xlabel('归一化空间频率');
ylabel('归一化多普勒频率');
title('STAP最优空时权重的二维响应');

% 显示 STAP 最优权重的三维响应
figure;
mesh(SpatialFreqGrid_normalized, fdGrid_Normalized, 10*log10(max(abs(Y).^2, eps)))
colorbar;
xlabel('归一化空间频率');
ylabel('归一化多普勒频率');
zlabel('响应功率 / dB');
title('STAP最优空时权重的三维响应');


%% (9) SINR 损失，定义参见文献 [2] Eq. (120)
sintheta = linspace(-1, 1, 181);
SpatialFreqGrid_normalized = d./lambda*sintheta; % d/lambda = 0.5
% Doppler 扫描轴独立覆盖未模糊区间 [-0.5, 0.5]
fdGrid_Normalized = linspace(-0.5, 0.5, length(SpatialFreqGrid_normalized));
a_Grid = exp(-1j*2*pi*[0: N - 1].'*SpatialFreqGrid_normalized);
b_Grid = exp(-1j*2*pi*[0:M - 1].'*fdGrid_Normalized);

SINR_loss = zeros(length(fdGrid_Normalized), length(SpatialFreqGrid_normalized));

for iSpatialFreq = 1:length(SpatialFreqGrid_normalized)
    for j_fdNormalized = 1:length(fdGrid_Normalized)

        % 当前测试空时导向矢量
        v = kron(b_Grid(:, j_fdNormalized), a_Grid(:, iSpatialFreq));

        % 干扰环境中的最大输出 SINR 因子：
        %   v^H R^{-1} v
        % 使用 R\v 代替显式求逆。
        SINR_current = real(v'*(R\v));

        % 仅有白噪声时的最优 SNR 因子：
        %   (v^H v)/noisePower
        % 这样即使 noisePower ~= 1，SINR loss 的定义仍然正确。
        SNRopt = real(v'*v)/noisePower;

        % SINR 损失 = 干扰环境最大输出 SINR / 白噪声条件最优 SNR
        SINR_loss(j_fdNormalized, iSpatialFreq) = SINR_current/SNRopt;
    end
end

% 取第 91 个空间频率切片。
% 由于此处共有 181 个空间频率点，第 91 点对应归一化空间频率 0。
% 此时矩阵行方向变化的是 Doppler，因此横坐标必须使用 fdGrid_Normalized。
figure;
plot(fdGrid_Normalized, 10*log10(max(real(SINR_loss(:, 91)), eps)));
xlabel('归一化多普勒频率');
ylabel('SINR损失 / dB');
title('零空间频率处的SINR损失');
