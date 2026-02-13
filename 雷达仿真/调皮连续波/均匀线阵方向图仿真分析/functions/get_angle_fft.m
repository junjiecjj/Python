function ret = get_angle_fft(bf, X)
% GET_ANGLE_FFT 计算角度谱
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 对阵列信号进行FFT得到角度谱
% 版本: 1.0
% 输入参数:
%   bf - 波束形成DOA估计结构体
%   X - 阵列信号向量或矩阵
% 输出参数:
%   ret - 角度谱

    if bf.array_dimension == 1
        % 一维FFT
        if isscalar(bf.angle_fft_size)
            fft_sz = bf.angle_fft_size;
        else
            fft_sz = bf.angle_fft_size(1);
        end
        ret = fftshift(fft(X, fft_sz), 1);
        
    elseif bf.array_dimension == 2
        % 二维FFT
        if isscalar(bf.angle_fft_size)
            fft_sz = [bf.angle_fft_size, bf.angle_fft_size];
        else
            fft_sz = bf.angle_fft_size;
        end
        % 使用 fftshift 对所有维度进行 shift（兼容旧版本MATLAB）
        ret = fftshift(fft2(X, fft_sz(1), fft_sz(2)));
        
    else
        ret = [];
    end
end