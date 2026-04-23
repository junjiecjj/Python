

clc;
clear all;
close all;
% addpath('./functions');

rng(42); 


M = 10;
Theta0 = [0, 5, 10];
N = 256;

% 角度扫描
theta_grid = -90:0.1:90;
% 导向矢量函数 (均匀线阵，半波长间距)
afun = @(theta) exp(-1j * pi * (0:M-1)' * sind(theta));


Gtr1 = zeros(3, length(theta_grid));
Gtr0 = zeros(3, length(theta_grid));

for i = 1:length(Theta0)
    theta0 = Theta0(i);
    a0 = afun(theta0);
    for k = 1:length(theta_grid)
        ak = afun(theta_grid(k));
        Gtr1(i, k) = N * abs(a0' * ones(M, 1))^2 * abs(ak' * a0)^2 / M;
        Gtr0(i, k) = N * abs(ak' * a0)^4 / M;
    end
end

%% 
width = 8;%设置图宽，这个不用改
height = 7*0.75;%设置图高，这个不用改
fontsize = 18;%设置图中字体大小
linewidth = 2;%设置线宽，一般大小为2，好看些。1是默认大小
marksize = 12;%标记的大小，按照个人喜好设置

%-------------------Figure1----------------------
h1 = figure(1);
%这是用于裁剪figure的。需要把fig.m文件放在一个文件夹中
fig(h1, 'units','inches','width',width, 'height', height, 'font','Times New Roman','fontsize',fontsize);
% axes1 = axes('Parent',h1,...
%     'Position',[0.091785714285714 0.1099604743083 0.390357142857143 0.858418972332016]);%left bottom width heigh
%颜色集合，这是默认的八种颜色，颜色的数量可以更改0.101785714285714 0.1699604743083 0.380357142857143 0.798418972332016
ColorSet = [
1       0       0      % 红色
0       1       0      % 绿色
0       0       1      % 蓝色
1       0       1      % 洋红色
0       1       1      % 青色
1       0.5     0      % 橙色
0.5     0       0.5  % 紫色
0.6     0.3     0   % 棕色
0       0.5     0   % 深绿色

0       0       0
0       0       0
1       0.75    0.8 % 粉色
0.5     0       0   % 深红色
0.5     0.5     0   % 橄榄绿
0       0       0.5 % 深蓝色
0       0       0   % 黑色
0.5     0.5     0.5 % 灰色
    ];
%设置循环使用的颜色集合
set(gcf, 'DefaultAxesColorOrder', ColorSet);

plot(theta_grid, pow2db(Gtr1(1,:)/max(Gtr1, [], 'all')), '-', 'LineWidth', 1.5); hold on;
plot(theta_grid, pow2db(Gtr1(2,:)/max(Gtr1, [], 'all')), '-', 'LineWidth', 1.5); hold on;
plot(theta_grid, pow2db(Gtr1(3,:)/max(Gtr1, [], 'all')), '-', 'LineWidth', 1.5); hold on;

xlabel('\theta (degrees)');
ylabel('G(\theta) (dB)');
legend('\theta = 0', '\theta = 5', '\theta = 10');
grid on; 
xlim([-40, 40]);
ylim([-60, 0]);


%-------------------Figure1----------------------
h2 = figure(2);
%这是用于裁剪figure的。需要把fig.m文件放在一个文件夹中
fig(h2, 'units','inches','width',width, 'height', height, 'font','Times New Roman','fontsize',fontsize);
% axes1 = axes('Parent',h1,...
%     'Position',[0.091785714285714 0.1099604743083 0.390357142857143 0.858418972332016]);%left bottom width heigh
%颜色集合，这是默认的八种颜色，颜色的数量可以更改0.101785714285714 0.1699604743083 0.380357142857143 0.798418972332016
ColorSet = [
1       0       0      % 红色
0       1       0      % 绿色
0       0       1      % 蓝色
1       0       1      % 洋红色
0       1       1      % 青色
1       0.5     0      % 橙色
0.5     0       0.5  % 紫色
0.6     0.3     0   % 棕色
0       0.5     0   % 深绿色

0       0       0
0       0       0
1       0.75    0.8 % 粉色
0.5     0       0   % 深红色
0.5     0.5     0   % 橄榄绿
0       0       0.5 % 深蓝色
0       0       0   % 黑色
0.5     0.5     0.5 % 灰色
    ];
%设置循环使用的颜色集合
set(gcf, 'DefaultAxesColorOrder', ColorSet);

plot(theta_grid, pow2db(Gtr0(1,:)/max(Gtr0, [], 'all')), '-', 'LineWidth', 1.5); hold on;
plot(theta_grid, pow2db(Gtr0(2,:)/max(Gtr0, [], 'all')), '-', 'LineWidth', 1.5); hold on;
plot(theta_grid, pow2db(Gtr0(3,:)/max(Gtr0, [], 'all')), '-', 'LineWidth', 1.5); hold on;

xlabel('\theta (degrees)');
ylabel('G(\theta) (dB)');
legend('\theta = 0', '\theta = 5', '\theta = 10');
grid on; 
xlim([-40, 40]);
ylim([-60, 0]);




































































































































