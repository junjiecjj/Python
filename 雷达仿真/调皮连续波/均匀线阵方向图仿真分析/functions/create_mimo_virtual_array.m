function mva = create_mimo_virtual_array(mc)
% CREATE_MIMO_VIRTUAL_ARRAY 创建MIMO虚拟阵列结构体
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 创建包含虚拟阵列信息的结构体，自动判断阵列类型
% 版本: 1.0
% 输入参数:
%   mc - MIMO配置结构体
% 输出参数:
%   mva - MIMO虚拟阵列结构体

    % 获取虚拟阵列位置
    va_arr = mc.va;
    
    % 检查阵列结构
    has_neg = min(va_arr(:));
    has_y = any(va_arr(:,:,2) ~= 0, 'all');
    is_supported = (has_neg >= 0) && ~has_y;
    
    if ~is_supported
        error('仅支持x方向线阵或xz平面阵列');
    end
    
    % 计算掩码
    nx_max = max(va_arr(:,:,1), [], 'all');
    nz_max = max(va_arr(:,:,3), [], 'all');
    mva.va_mask = zeros(nz_max + 1, nx_max + 1);
    
    has_z_axis = false;
    index_vec = [];
    index_mat = {[], []};
    
    for k = 1:size(va_arr, 1)
        for l = 1:size(va_arr, 2)
            x = va_arr(k, l, 1) + 1;  % MATLAB索引从1开始
            z = va_arr(k, l, 3) + 1;
            
            if mva.va_mask(z, x) == 0
                mva.va_mask(z, x) = 1;
                % 使用行优先(C-order)索引计算，与Python保持一致
                index_vec = [index_vec, sub2ind(size(mva.va_mask), z, x)];
                index_mat{1} = [index_mat{1}, z];
                index_mat{2} = [index_mat{2}, x];
            end
            
            if z > 1
                has_z_axis = true;
            end
        end
    end
    
    mva.index_vec = index_vec;
    mva.index_mat = index_mat;
    mva.is_uniform = numel(mva.va_mask) == sum(mva.va_mask(:));
    
    if ~has_z_axis
        % 一维阵列(ULA)
        type_str = 'uniform';
        if ~mva.is_uniform
            type_str = 'general';
        end
        fprintf('阵列类型: %s 线阵\n', type_str);
        mva.array_dimension = 1;
        mva.la = create_ula(size(mva.va_mask, 2), mc.d);
    else
        % 二维阵列(URA)
        type_str = 'xz uniform rectangular';
        if ~mva.is_uniform
            type_str = 'general xz planar';
        end
        fprintf('阵列类型: %s 阵列\n', type_str);
        mva.array_dimension = 2;
        mva.pa = create_ura(size(mva.va_mask), mc.d);
    end
end