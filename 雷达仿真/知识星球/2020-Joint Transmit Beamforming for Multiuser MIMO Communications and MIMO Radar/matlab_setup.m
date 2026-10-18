% ========================================
% MATLAB 环境一键配置脚本
% 在 MATLAB 命令窗口中运行：run('matlab_setup.m')
% ========================================

clc;
fprintf('========================================\n');
fprintf('  空地协同 MIMO ISAC 抗干扰波束赋形\n');
fprintf('  MATLAB 环境配置\n');
fprintf('========================================\n\n');

%% 1. 添加代码目录到 MATLAB 永久路径
codeDir = fileparts(mfilename('fullpath'));
addpath(genpath(codeDir));
try
    savepath;
    fprintf('[OK] 代码路径已添加并保存: %s\n', codeDir);
catch
    fprintf('[WARN] 路径添加成功但保存失败（可能需要管理员权限）\n');
    fprintf('       每次启动 MATLAB 请手动运行: addpath(genpath(''%s''))\n', codeDir);
end

%% 2. 配置 CVX + SeDuMi
cvxDir = fullfile(getenv('USERPROFILE'), 'Documents', 'MATLAB', 'cvx');
if exist(fullfile(cvxDir, 'cvx_setup.m'), 'file')
    fprintf('\n[OK] CVX 已安装在: %s\n', cvxDir);

    % 运行 cvx_setup
    oldDir = pwd;
    cd(cvxDir);
    try
        cvx_setup;
        fprintf('[OK] CVX + SeDuMi 配置完成\n');
    catch ME
        fprintf('[WARN] cvx_setup 运行出错: %s\n', ME.message);
        fprintf('       请在 MATLAB 中手动运行: cd(''%s''); cvx_setup\n', cvxDir);
    end
    cd(oldDir);
else
    fprintf('\n[WARN] CVX 未找到于: %s\n', cvxDir);

    % 尝试在代码目录中找
    cvxLocal = fullfile(codeDir, 'cvx', 'cvx_setup.m');
    if exist(cvxLocal, 'file')
        oldDir = pwd;
        cd(fullfile(codeDir, 'cvx'));
        cvx_setup;
        cd(oldDir);
        fprintf('[OK] CVX 从本地目录配置\n');
    else
        fprintf('[TODO] 请下载 CVX: http://cvxr.com/cvx/download/\n');
        fprintf('       解压后将 cvx_setup.m 所在目录添加到路径\n');
    end
end

%% 3. 验证环境
fprintf('\n========================================\n');
fprintf('  环境验证\n');
fprintf('========================================\n');

% 验证 SeDuMi
try
    cvx_solver sedumi;
    fprintf('[OK] SeDuMi 求解器可用\n');
catch
    fprintf('[FAIL] SeDuMi 求解器不可用\n');
end

% 验证 SDPT3
try
    cvx_solver sdpt3;
    fprintf('[OK] SDPT3 求解器可用\n');
catch
    fprintf('[WARN] SDPT3 求解器不可用（SeDuMi 作为默认即可）\n');
end

cvx_solver sedumi;  % 设回默认

% 验证代码文件
mFiles = dir(fullfile(codeDir, '*.m'));
fprintf('\n代码文件 (%d 个):\n', length(mFiles));
for i = 1:length(mFiles)
    fprintf('  %s\n', mFiles(i).name);
end

% 快速语法检查
fprintf('\n函数语法检查:\n');
funcFiles = {'ULA_steering_vector', 'nearestSPD', 'eval_null_depth', ...
             'rician_channel', 'QPSK_mapper', 'QPSK_demodulation', ...
             'INR_func', 'SUM_RATE_func'};
for i = 1:length(funcFiles)
    try
        nargin(funcFiles{i});
        fprintf('  [OK] %s\n', funcFiles{i});
    catch
        fprintf('  [FAIL] %s 不存在或无法访问\n', funcFiles{i});
    end
end

fprintf('\n========================================\n');
fprintf('  配置完成！\n');
fprintf('  运行 main_anti_jamming 开始仿真\n');
fprintf('========================================\n');
