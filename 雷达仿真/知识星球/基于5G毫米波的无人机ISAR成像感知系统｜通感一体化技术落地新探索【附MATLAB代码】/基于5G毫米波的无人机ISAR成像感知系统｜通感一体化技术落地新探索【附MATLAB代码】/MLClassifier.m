%% MATLAB机器学习分类系统
% =============================================
% 功能：
% 1. 特征提取（HOG、LBP、统计特征）
% 2. 分类器（SVM、决策树、神经网络）
% 3. 性能评估
% 4. 可视化
% =============================================

classdef MLClassifier < handle
    
    properties
        features        % 特征矩阵
        labels          % 标签
        model           % 训练好的模型
        model_type      % 模型类型
        metrics         % 性能指标
    end
    
    methods
        function obj = MLClassifier(model_type)
            % 构造函数
            % model_type: 'svm', 'tree', 'nn', 'ensemble'
            obj.model_type = model_type;
            obj.metrics = struct();
        end
        
        function features = getHOGFeatures(obj, image)
            % 提取HOG特征（方向梯度直方图）
            % 注意：需要 Computer Vision Toolbox

            % 确保图像是灰度图
            if size(image, 3) == 3
                image = rgb2gray(image);
            end

            % 调用内置函数 extractHOGFeatures
            features = extractHOGFeatures(image, 'CellSize', [8 8]);
        end

        function features = getLBPFeatures(obj, image)
            % 提取LBP特征（局部二值模式）
            % 注意：需要 Computer Vision Toolbox

            if size(image, 3) == 3
                image = rgb2gray(image);
            end

            % 调用内置函数 extractLBPFeatures
            features = extractLBPFeatures(image, 'Upright', false);
        end

        function features = getStatisticalFeatures(obj, image)
            % 提取统计特征

            if size(image, 3) == 3
                image = rgb2gray(image);
            end

            % 基础统计特征
            mean_val = mean(image(:));
            std_val = std(image(:));
            max_val = max(image(:));
            min_val = min(image(:));

            % 方差（替代 moment，不依赖 Statistics Toolbox）
            mu = var(double(image(:)));

            % 纹理特征
            try
                glcm = graycomatrix(uint8(image * 255));
                stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
                features = [mean_val, std_val, max_val, min_val, mu, ...
                           stats.Contrast, stats.Correlation, ...
                           stats.Energy, stats.Homogeneity];
            catch
                % 无 Image Processing Toolbox，使用替代特征
                grad_energy = sum(abs(diff(double(image(:)))));
                h_grad = sum(abs(diff(double(image), 1, 2)), 'all');
                features = [mean_val, std_val, max_val, min_val, mu, ...
                           grad_energy, h_grad];
            end
        end

        function features = extractAllFeatures(obj, image)
            % 提取所有特征的组合

            hog_feat = obj.getHOGFeatures(image);
            lbp_feat = obj.getLBPFeatures(image);
            stat_feat = obj.getStatisticalFeatures(image);

            % 拼接特征
            features = [hog_feat, lbp_feat, stat_feat];
        end
        
        function prepareDataset(obj, images, labels, feature_type)
            % 准备数据集
            % feature_type: 'hog', 'lbp', 'stat', 'all'
            
            fprintf('正在提取特征（方法：%s）...\n', feature_type);
            
            num_samples = size(images, 4);
            
            % 提取第一张图像的特征来确定特征维度
            if strcmp(feature_type, 'all')
                first_features = obj.extractAllFeatures(images(:,:,:,1));
            elseif strcmp(feature_type, 'hog')
                first_features = obj.getHOGFeatures(images(:,:,:,1));
            elseif strcmp(feature_type, 'lbp')
                first_features = obj.getLBPFeatures(images(:,:,:,1));
            else
                first_features = obj.getStatisticalFeatures(images(:,:,:,1));
            end
            
            feature_dim = length(first_features);
            obj.features = zeros(num_samples, feature_dim);
            obj.features(1, :) = first_features;
            
            % 提取所有图像的特征
            for i = 2:num_samples
                if strcmp(feature_type, 'all')
                    obj.features(i, :) = obj.extractAllFeatures(images(:,:,:,i));
                elseif strcmp(feature_type, 'hog')
                    obj.features(i, :) = obj.getHOGFeatures(images(:,:,:,i));
                elseif strcmp(feature_type, 'lbp')
                    obj.features(i, :) = obj.getLBPFeatures(images(:,:,:,i));
                else
                    obj.features(i, :) = obj.getStatisticalFeatures(images(:,:,:,i));
                end
                
                if mod(i, 100) == 0
                    fprintf('  进度: %d/%d\n', i, num_samples);
                end
            end
            
            obj.labels = labels;
            
            fprintf('  特征提取完成！特征维度: %d\n', feature_dim);
        end
        
        function train(obj, train_features, train_labels)
            % 训练模型

            fprintf('正在训练模型（类型：%s）...\n', obj.model_type);

            try
                switch obj.model_type
                    case 'svm'
                        obj.model = fitcecoc(train_features, train_labels, ...
                                            'Learners', 'svm');
                    case 'tree'
                        obj.model = fitctree(train_features, train_labels, ...
                                            'MaxNumSplits', 100);
                    case 'nn'
                        obj.model = fitcnet(train_features, train_labels, ...
                                           'LayerSizes', [100, 50]);
                    case 'ensemble'
                        obj.model = TreeBagger(100, train_features, train_labels, ...
                                              'OOBPrediction', 'on', ...
                                              'Method', 'classification');
                    otherwise
                        error('未知的模型类型: %s', obj.model_type);
                end
            catch ME
                fprintf('  注意: %s\n', ME.message);
                fprintf('  使用 KNN 替代（不依赖工具箱）\n');
                obj.model_type = 'knn_fallback';
                obj.model = struct('train_features', train_features, ...
                                   'train_labels', train_labels, 'k', 5);
            end

            fprintf('  训练完成！\n');
        end
        
        function [pred_labels, scores] = predict(obj, test_features)
            % 预测

            if strcmp(obj.model_type, 'knn_fallback')
                % 手动KNN预测
                k = obj.model.k;
                n_test = size(test_features, 1);
                pred_labels = zeros(n_test, 1);
                scores = zeros(n_test, 1);
                for idx = 1:n_test
                    dists = sum((obj.model.train_features - test_features(idx,:)).^2, 2);
                    [~, sorted_idx] = sort(dists);
                    nearest_labels = obj.model.train_labels(sorted_idx(1:k));
                    pred_labels(idx) = mode(nearest_labels);
                    scores(idx) = sum(nearest_labels == pred_labels(idx)) / k;
                end
            elseif strcmp(obj.model_type, 'ensemble')
                [pred_labels, scores] = predict(obj.model, test_features);
                pred_labels = str2double(pred_labels);
            else
                [pred_labels, scores] = predict(obj.model, test_features);
            end
        end
        
        function accuracy = evaluate(obj, test_features, test_labels, class_names)
            % 评估模型
            
            fprintf('\n===== 模型评估 =====\n');
            
            % 预测
            [pred_labels, scores] = obj.predict(test_features);
            
            % 计算准确率
            accuracy = sum(pred_labels == test_labels) / length(test_labels) * 100;
            fprintf('准确率: %.2f%%\n', accuracy);
            obj.metrics.accuracy = accuracy;
            
            % 混淆矩阵（手动计算，不依赖 Statistics Toolbox）
            unique_labels = unique([test_labels; pred_labels]);
            n_labels = length(unique_labels);
            cm = zeros(n_labels, n_labels);
            for idx = 1:length(test_labels)
                row = find(unique_labels == test_labels(idx));
                col = find(unique_labels == pred_labels(idx));
                cm(row, col) = cm(row, col) + 1;
            end
            obj.metrics.confusion_matrix = cm;
            
            % 绘制混淆矩阵
            obj.plotConfusionMatrix(cm, class_names);
            
            % 计算每个类别的精确率、召回率
            num_classes = length(class_names);
            fprintf('\n各类别性能:\n');
            fprintf('%-15s %-10s %-10s %-10s\n', '类别', '精确率', '召回率', 'F1分数');
            fprintf('---------------------------------------------------\n');
            
            for i = 1:num_classes
                tp = cm(i, i);
                fp = sum(cm(:, i)) - tp;
                fn = sum(cm(i, :)) - tp;
                
                precision = tp / (tp + fp);
                recall = tp / (tp + fn);
                f1 = 2 * precision * recall / (precision + recall);
                
                fprintf('%-15s %.4f     %.4f     %.4f\n', ...
                       class_names{i}, precision, recall, f1);
            end
            
            fprintf('===========================\n\n');
        end
        
        function plotConfusionMatrix(obj, cm, class_names)
            % 绘制混淆矩阵
            
            figure('Color', 'w');
            
            % 归一化混淆矩阵
            cm_norm = cm ./ sum(cm, 2);
            
            % 绘制热图
            imagesc(cm_norm);
            colormap('hot');
            colorbar;
            
            % 添加数值标注
            [rows, cols] = size(cm);
            for i = 1:rows
                for j = 1:cols
                    text(j, i, sprintf('%d\n(%.2f%%)', cm(i,j), cm_norm(i,j)*100), ...
                        'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'middle', ...
                        'Color', 'white', ...
                        'FontSize', 10);
                end
            end
            
            % 设置坐标轴
            set(gca, 'XTick', 1:cols, 'XTickLabel', class_names);
            set(gca, 'YTick', 1:rows, 'YTickLabel', class_names);
            xlabel('预测类别', 'FontSize', 12);
            ylabel('真实类别', 'FontSize', 12);
            title('混淆矩阵', 'FontSize', 14);
            
            saveas(gcf, 'confusion_matrix_matlab.png');
        end
        
        function crossValidate(obj, features, labels, k)
            % K折交叉验证

            fprintf('正在进行%d折交叉验证...\n', k);

            N = size(features, 1);
            accuracies = zeros(k, 1);

            % 手动实现K折划分（不依赖 cvpartition）
            indices = crossvalind_manual(k, N);

            for i = 1:k
                test_idx = (indices == i);
                train_idx = ~test_idx;

                % 训练
                obj.train(features(train_idx, :), labels(train_idx));

                % 测试
                [pred, ~] = obj.predict(features(test_idx, :));
                accuracies(i) = sum(pred == labels(test_idx)) / sum(test_idx) * 100;

                fprintf('  Fold %d: %.2f%%\n', i, accuracies(i));
            end
            
            fprintf('平均准确率: %.2f%% (±%.2f%%)\n', ...
                   mean(accuracies), std(accuracies));
            
            obj.metrics.cv_mean = mean(accuracies);
            obj.metrics.cv_std = std(accuracies);
        end
        
        function saveModel(obj, filepath)
            % 保存模型
            model_struct.model = obj.model;
            model_struct.model_type = obj.model_type;
            model_struct.metrics = obj.metrics;
            save(filepath, 'model_struct');
            fprintf('模型已保存: %s\n', filepath);
        end
        
        function loadModel(obj, filepath)
            % 加载模型
            load(filepath, 'model_struct');
            obj.model = model_struct.model;
            obj.model_type = model_struct.model_type;
            obj.metrics = model_struct.metrics;
            fprintf('模型已加载: %s\n', filepath);
        end
    end
end

%% 辅助函数：生成合成数据
function [images, labels] = generateSyntheticData(num_samples, num_classes, image_size)
    % 生成合成ISAR数据
    
    fprintf('生成合成数据: %d 样本, %d 类别\n', num_samples, num_classes);
    
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

%% 使用示例
function demo_ml_classification()
    % 1. 生成数据
    fprintf('===== 步骤1: 生成数据 =====\n');
    [images, labels] = generateSyntheticData(1000, 5, 128);
    
    % 2. 划分数据集
    fprintf('\n===== 步骤2: 划分数据集 =====\n');
    split_ratio = 0.7;
    num_train = round(split_ratio * length(labels));
    
    train_images = images(:, :, :, 1:num_train);
    train_labels = labels(1:num_train);
    test_images = images(:, :, :, num_train+1:end);
    test_labels = labels(num_train+1:end);
    
    fprintf('训练集: %d 样本\n', num_train);
    fprintf('测试集: %d 样本\n', length(test_labels));
    
    % 3. 创建分类器并提取特征
    fprintf('\n===== 步骤3: 特征提取 =====\n');
    classifier = MLClassifier('svm');
    classifier.prepareDataset(images, labels, 'stat');  % 使用统计特征（速度快）
    
    train_features = classifier.features(1:num_train, :);
    test_features = classifier.features(num_train+1:end, :);
    
    % 4. 训练
    fprintf('\n===== 步骤4: 训练模型 =====\n');
    classifier.train(train_features, train_labels);
    
    % 5. 评估
    fprintf('\n===== 步骤5: 评估模型 =====\n');
    class_names = {'小型', '中型', '大型', '固定翼', '旋翼'};
    accuracy = classifier.evaluate(test_features, test_labels, class_names);
    
    % 6. 交叉验证
    fprintf('\n===== 步骤6: 交叉验证 =====\n');
    classifier.crossValidate(classifier.features, labels, 5);
    
    % 7. 保存模型
    fprintf('\n===== 步骤7: 保存模型 =====\n');
    classifier.saveModel('drone_classifier.mat');
    
    fprintf('\n演示完成！\n');
end

function indices = crossvalind_manual(k, N)
    % 手动实现K折交叉验证索引划分
    indices = zeros(N, 1);
    perm = randperm(N);
    fold_size = floor(N / k);
    for i = 1:k
        if i <= mod(N, k)
            start_idx = (i-1)*(fold_size+1) + 1;
            end_idx = i*(fold_size+1);
        else
            start_idx = mod(N,k)*(fold_size+1) + (i-mod(N,k)-1)*fold_size + 1;
            end_idx = start_idx + fold_size - 1;
        end
        indices(perm(start_idx:end_idx)) = i;
    end
end
