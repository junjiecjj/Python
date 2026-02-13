function dfft = get_2d_fft(data_cube, use_2d_fft)
% GET_2D_FFT 二维FFT处理
% 作者: Radar Engineer
% 时间: 2026-02-13
% 功能: 对雷达数据立方体进行距离-多普勒FFT处理
% 版本: 1.2
% 输入参数:
%   data_cube - 数据立方体 [... × chirps × ADC]
%   use_2d_fft - 是否使用2D FFT，默认false
% 输出参数:
%   dfft - 距离-多普勒图

    if nargin < 2
        use_2d_fft = false;
    end
    
    % 获取最后两个维度的大小
    nd = ndims(data_cube);
    chirp_size = size(data_cube, nd - 1);
    adc_size = size(data_cube, nd);
    
    % 检查维度是否有效
    if chirp_size <= 0 || adc_size <= 0
        error('get_2d_fft: 数据维度无效, chirp_size=%d, adc_size=%d', chirp_size, adc_size);
    end
    
    % 与Python保持一致：获取FFT大小
    % Python: szl = get_fft_sizes(data_cube.shape[-2:]) -> (chirps, adc)
    szl = get_fft_sizes([chirp_size, adc_size]);
    adc_fft_size = szl{2};
    chirp_fft_size = szl{1};
    
    if use_2d_fft
        % 与Python保持一致：np.fft.fft2(data_cube, s=szl[::-1], axes=(-1, -2))
        % 注意：Python的fft2参数s是(adc_fft_size, chirp_fft_size)
        % 而MATLAB的fft2语法是fft2(X, m, n)，其中m是行数，n是列数
        % 对于数据立方体 [... × chirps × ADC]，最后两维是(chirps, ADC)
        % 所以我们需要对最后两维进行FFT，保持维度顺序
        dfft = fft2(data_cube, chirp_fft_size, adc_fft_size);
        % 对多普勒维度（chirp）进行fftshift
        dfft = fftshift(dfft, nd - 1);
    else
        % 分步处理，便于添加窗函数
        % 对最后一维(ADC)进行FFT
        rfft = fft(data_cube, adc_fft_size, nd);
        % 对倒数第二维(chirp)进行FFT
        dfft = fft(rfft, chirp_fft_size, nd - 1);
        % 多普勒维fftshift
        dfft = fftshift(dfft, nd - 1);
    end
end