%% MATLAB机器学习分类 - 演示脚本
% =============================================
% 功能：运行机器学习分类演示
% 使用：直接运行 Demo_ML_MATLAB
% =============================================

clear; clc; close all;

fprintf('====================================\n');
fprintf('  MATLAB机器学习分类演示\n');
fprintf('====================================\n\n');

%% 1. 生成合成数据
fprintf('步骤1: 生成合成数据...\n');
num_samples = 1000;
num_classes = 5;
image_size = 128;

[images, labels] = generateSyntheticData(num_samples, num_classes, image_size);
fprintf('  ✓ 已生成 %d 个样本\n\n', num_samples);

%% 2. 划分数据集
fprintf('步骤2: 划分数据集...\n');
split_ratio = 0.7;
num_train = round(split_ratio * length(labels));

train_images = images(:, :, :, 1:num_train);
train_labels = labels(1:num_train);
test_images = images(:, :, :, num_train+1:end);
test_labels = labels(num_train+1:end);

fprintf('  训练集: %d 样本\n', num_train);
fprintf('  测试集: %d 样本\n\n', length(test_labels));

%% 3. 创建分类器并提取特征
fprintf('步骤3: 提取特征（使用统计特征）...\n');
classifier = MLClassifier('svm');
classifier.prepareDataset(images, labels, 'stat');

train_features = classifier.features(1:num_train, :);
test_features = classifier.features(num_train+1:end, :);
fprintf('  ✓ 特征提取完成\n\n');

%% 4. 训练模型
fprintf('步骤4: 训练SVM模型...\n');
classifier.train(train_features, train_labels);
fprintf('  ✓ 训练完成\n\n');

%% 5. 评估模型
fprintf('步骤5: 评估模型...\n');
class_names = {'小型', '中型', '大型', '固定翼', '旋翼'};
accuracy = classifier.evaluate(test_features, test_labels, class_names);

%% 6. 交叉验证
fprintf('\n步骤6: 5折交叉验证...\n');
classifier.crossValidate(classifier.features, labels, 5);

%% 7. 保存模型
fprintf('\n步骤7: 保存模型...\n');
classifier.saveModel('drone_classifier.mat');
fprintf('  ✓ 模型已保存为 drone_classifier.mat\n\n');

fprintf('====================================\n');
fprintf('  演示完成！\n');
fprintf('====================================\n');


%% 辅助函数：生成合成数据
function [images, labels] = generateSyntheticData(num_samples, num_classes, image_size)
    % 生成合成ISAR数据
    
    images = zeros(image_size, image_size, 1, num_samples);
    labels = zeros(num_samples, 1);
    
    for i = 1:num_samples
        % 随机选择类别
        label = randi(num_classes);
        labels(i) = label;
        
        % 生成图像
        img = zeros(image_size, image_size);
        
        if label == 1  % 小型无人机
            [X, Y] = meshgrid(linspace(-1,1,image_size), linspace(-1,1,image_size));
            img = exp(-(X.^2 + Y.^2) / 0.1);
            
        elseif label == 2  % 中型无人机
            img(40:90, 40:90) = 1.0;
            
        elseif label == 3  % 大型无人机
            img(20:110, 20:110) = 0.8;
            img(60:70, :) = 1.0;
            
        elseif label == 4  % 固定翼
            img(60:70, :) = 1.0;
            img(:, 60:70) = 0.5;
            
        else  % 旋翼机
            for angle = linspace(0, 2*pi, 4)
                x = round(64 + 30*cos(angle));
                y = round(64 + 30*sin(angle));
                x = max(1, min(image_size, x));
                y = max(1, min(image_size, y));
                img(max(1,y-10):min(image_size,y+10), ...
                   max(1,x-10):min(image_size,x+10)) = 0.8;
            end
        end
        
        % 添加噪声
        img = img + randn(image_size, image_size) * 0.1;
        img = max(0, min(1, img));
        
        images(:, :, 1, i) = img;
    end
end
