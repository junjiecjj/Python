function [images, labels] = generateSyntheticData(num_samples, num_classes, image_size)
%% generateSyntheticData - 生成合成ISAR数据
% 供 Demo_ML_MATLAB.m 和其他脚本调用
%
% 输入：
%   num_samples  - 样本数量
%   num_classes  - 类别数量
%   image_size   - 图像尺寸（正方形边长）
%
% 输出：
%   images - image_size x image_size x 1 x num_samples 的图像数组
%   labels - num_samples x 1 的标签向量

    fprintf('生成合成数据: %d 样本, %d 类别\n', num_samples, num_classes);

    images = zeros(image_size, image_size, 1, num_samples);
    labels = zeros(num_samples, 1);

    for i = 1:num_samples
        label = randi(num_classes);
        labels(i) = label;

        img = zeros(image_size, image_size);

        if label == 1  % 小型无人机
            [X, Y] = meshgrid(linspace(-1,1,image_size), linspace(-1,1,image_size));
            img = exp(-(X.^2 + Y.^2) / 0.1);

        elseif label == 2  % 中型无人机
            r1 = round(image_size*0.3); r2 = round(image_size*0.7);
            img(r1:r2, r1:r2) = 1.0;

        elseif label == 3  % 大型无人机
            r1 = round(image_size*0.15); r2 = round(image_size*0.85);
            mid1 = round(image_size*0.45); mid2 = round(image_size*0.55);
            img(r1:r2, r1:r2) = 0.8;
            img(mid1:mid2, :) = 1.0;

        elseif label == 4  % 固定翼
            mid1 = round(image_size*0.45); mid2 = round(image_size*0.55);
            img(mid1:mid2, :) = 1.0;
            img(:, mid1:mid2) = 0.5;

        else  % 旋翼机
            center = round(image_size/2);
            radius = round(image_size*0.23);
            arm = round(image_size*0.08);
            for angle = linspace(0, 2*pi, 4)
                x = round(center + radius*cos(angle));
                y = round(center + radius*sin(angle));
                x = max(1, min(image_size, x));
                y = max(1, min(image_size, y));
                img(max(1,y-arm):min(image_size,y+arm), ...
                   max(1,x-arm):min(image_size,x+arm)) = 0.8;
            end
        end

        % 添加噪声
        img = img + randn(image_size, image_size) * 0.1;
        img = max(0, min(1, img));

        images(:, :, 1, i) = img;
    end
end
