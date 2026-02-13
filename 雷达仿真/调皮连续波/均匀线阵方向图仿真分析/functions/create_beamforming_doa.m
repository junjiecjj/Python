function bf = create_beamforming_doa(mva, angle_fft_size)
% CREATE_BEAMFORMING_DOA 创建波束形成DOA估计结构体
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 创建包含波束形成参数的DOA估计结构体
% 版本: 1.0
% 输入参数:
%   mva            - MIMO虚拟阵列结构体
%   angle_fft_size - FFT点数，默认256
% 输出参数:
%   bf - 波束形成DOA估计结构体

    if nargin < 2
        angle_fft_size = 256;
    end
    
    bf.angle_fft_size = angle_fft_size;
    bf.array_dimension = get_array_dimension(mva);
    bf.angle_bins = get_fft_angle_bins(mva, angle_fft_size);
end