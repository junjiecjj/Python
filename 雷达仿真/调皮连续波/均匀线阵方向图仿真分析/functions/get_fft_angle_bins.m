function angle_bins = get_fft_angle_bins(mva, angle_fft_size)
% GET_FFT_ANGLE_BINS 获取FFT角度采样点
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 根据阵列维度获取相应的FFT角度采样点
% 版本: 1.0
% 输入参数:
%   mva - MIMO虚拟阵列结构体
%   angle_fft_size - FFT点数
% 输出参数:
%   angle_bins - 角度采样点

    if mva.array_dimension == 1
        if isscalar(angle_fft_size)
            fft_sz = angle_fft_size;
        else
            fft_sz = angle_fft_size(1);
        end
        angle_bins = get_angle_bins(mva.la, mva.la.d, fft_sz);
    else
        if isscalar(angle_fft_size)
            fft_sz = [angle_fft_size, angle_fft_size];
        else
            fft_sz = angle_fft_size;
        end
        [theta_bins, phi_bins] = get_angle_bins_2d(mva.pa, mva.pa.d, fft_sz);
        angle_bins = {theta_bins, phi_bins};
    end
end