clc;
%% 设置参数
lambda = 1.25e-2;
pos = [[0; 1; 0], [0; 5; 0], [0; 2; 0], [0; 3; 0]] * 0.5 * lambda;
ang = [[10; 0], [20; 10], [30; 20], [0; 30]];

%% 分别调用自写的函数和matlab的函数
my_sv = steering_vector(pos, ang, lambda);
matlab_sv = steervec(pos / lambda, ang);

%% 比较结果
matlab_sv
my_sv
diff = mean(abs((my_sv - matlab_sv) ./ matlab_sv), 'all')
