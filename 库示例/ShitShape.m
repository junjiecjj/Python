
clear;
clc;
close all;
%% icecream 1
[X, Y, Z] = shitshape();
surf(X, Y, Z, 'FaceColor', [0.4 0.2 0], 'EdgeColor', 'none');
axis equal off; camlight headlight; lighting gouraud;
view(-60, 10);

%% icecream 2
% 绘制shit
figure;
axis equal off; hold on; camlight headlight; lighting gouraud; view(3)
[X, Y, Z] = shitshape();
surf(X + .1, Y, Z + 1.06, 'FaceColor', [0.4 0.2 0], 'EdgeColor', 'none');
surf(X./1.3, Y./1.3, -Z + 1.1, 'FaceColor', [0.4 0.2 0], 'EdgeColor', 'none');

% 甜筒数据构造
N = 500; n = 30;
[X, Y] = meshgrid(linspace(-1, 1, N));
Z = X.*0;
Z(1:n:N, :) = .01; Z(:, 1:n:N) = .01;
Z(2:n:N, :) = .01; Z(:, 2:n:N) = .01;
Z(X.^2 + Y.^2 > 1) = nan;
R_x = @(theta)[1, 0, 0; 
       0, cos(theta), -sin(theta);
       0, sin(theta),  cos(theta)];
R_y = @(theta)[ 
       cos(theta), 0, -sin(theta);
       0, 1, 0;
       sin(theta), 0,  cos(theta)];
nXYZ = [X(:), Y(:), Z(:)]*R_x(-pi/2.9);
nX = nXYZ(:,1); nY = nXYZ(:,2); nZ = nXYZ(:,3);
nZ = nZ - min(nZ); nY = nY - min(nY);
nT = nX ./(nZ/2.5); nR = nY;
nX = cos(nT).*nR; nY = sin(nT).*nR;
SHdl = surf(reshape(nX, N, []), reshape(nY, N, []), reshape(nZ, N, []), ...
    'EdgeColor','none', 'FaceColor',[228,200,142]./255);
material(SHdl, 'dull')

% 绘制饼干
[X, Y, Z] = cylinder(0.04, 100); N = size(X, 1);
nXYZ = [X(:), Y(:), Z(:)]*R_x(pi/5);
nX = nXYZ(:,1); nY = nXYZ(:,2); nZ = nXYZ(:,3);
surf(reshape(nX, N, []), reshape(nY, N, []), reshape(nZ, N, []) + 1.3,...
    'FaceColor', [0.8 0.6 0.4], 'EdgeColor', 'none');
nXYZ = [X(:), Y(:), Z(:)]*R_x(pi/6);
nX = nXYZ(:,1); nY = nXYZ(:,2); nZ = nXYZ(:,3);
nXYZ = [nX(:), nY(:), nZ(:)]*R_y(pi/7);
nX = nXYZ(:,1); nY = nXYZ(:,2); nZ = nXYZ(:,3);
surf(reshape(nX, N, []), reshape(nY, N, []), reshape(nZ, N, []) + 1.2,...
    'FaceColor', [0.8 0.6 0.4], 'EdgeColor', 'none');

%% icecream 3
%function shit3
figure('Units','normalized','Position',[.2,.1,.6,.7])
%曲面数据计算 ==============================================================
[Xa, Ya, Za] = shitshape();
% 百合花部分 ---------------------------------------------------------------
rb=0:.01:1;
tb=linspace(0,2,151);
wb=rb'*((abs((1-mod(tb*5,2))))/2+.3);
xb=wb.*cospi(tb);
yb=wb.*sinpi(tb); 
zb=@(a)(-cospi(wb*a)+1).^.2;
Zb=zb(1.2);
%颜色映射表 ================================================================
colorList=[0.3300    0.3300    0.6900
    0.5300    0.4000    0.6800
    0.6800    0.4200    0.6300
    0.7800    0.4200    0.5700
    0.9100    0.4900    0.4700
    0.9600    0.7300    0.4400];
colorMapb=setColorByH(Zb,colorList.*.4+.6);
    function cMap=setColorByH(H,cList)
        X=(H-min(min(H)))./(max(max(H))-min(min(H)));
        xx=(0:size(cList,1)-1)./(size(cList,1)-1);
        y1=cList(:,1);y2=cList(:,2);y3=cList(:,3);
        cMap(:,:,1)=interp1(xx,y1,X,'linear');
        cMap(:,:,2)=interp1(xx,y2,X,'linear');
        cMap(:,:,3)=interp1(xx,y3,X,'linear');
    end
% 旋转函数预定义 ===========================================================
yaw_z=72*pi/180;
roll_x_1=pi/8;
roll_x_2=pi/9;
R_z_2=[cos(yaw_z)  , -sin(yaw_z)  , 0; sin(yaw_z)  , cos(yaw_z)  , 0; 0, 0, 1];
R_z_1=[cos(yaw_z/2), -sin(yaw_z/2), 0; sin(yaw_z/2), cos(yaw_z/2), 0; 0, 0, 1];
R_z_3=[cos(yaw_z/3), -sin(yaw_z/3), 0; sin(yaw_z/3), cos(yaw_z/3), 0; 0, 0, 1];
R_x_1=[1, 0, 0; 0, cos(roll_x_1), -sin(roll_x_1); 0, sin(roll_x_1), cos(roll_x_1)];
R_x_2=[1, 0, 0; 0, cos(roll_x_2), -sin(roll_x_2); 0, sin(roll_x_2), cos(roll_x_2)];
    function [nX,nY,nZ]=rotateXYZ(X,Y,Z,R)
        nX=zeros(size(X)); nY=zeros(size(Y)); nZ=zeros(size(Z));
        for i=1:size(X,1)
            for j=1:size(X,2)
                v=[X(i,j);Y(i,j);Z(i,j)];
                nv=R*v; nX(i,j)=nv(1); nY(i,j)=nv(2); nZ(i,j)=nv(3);
            end
        end
    end
% 绘制花杆函数预定义 ========================================================
    function drawStraw(X,Y,Z)
        [m,n]=find(Z==min(min(Z)));
        m=m(1);n=n(1);
        x1=X(m,n);y1=Y(m,n);z1=Z(m,n)+.03;
        xx=[x1,0,(x1.*cos(pi/3)-y1.*sin(pi/3))./3].';
        yy=[y1,0,(y1.*cos(pi/3)+x1.*sin(pi/3))./3].';
        zz=[z1,-.7,-1.5].';
        strawPnts=bezierCurve([xx,yy,zz],50);
        plot3(strawPnts(:,1),strawPnts(:,2),strawPnts(:,3),'Color',[88,130,126]./255,'LineWidth',2)
    end
% 贝塞尔函数 ---------------------------------------------------------------
    function pnts=bezierCurve(pnts,N)
        t=linspace(0,1,N);
        p=size(pnts,1)-1;
        coe1=factorial(p)./factorial(0:p)./factorial(p:-1:0);
        coe2=((t).^((0:p)')).*((1-t).^((p:-1:0)'));
        pnts=(pnts'*(coe1'.*coe2))';
    end
%曲面旋转及绘制 ============================================================
hold on
surface(Xa, Ya, Za + .7,'EdgeAlpha',0.05,...
    'EdgeColor',[0 0 0],'FaceColor',[0.4 0.2 0],'Tag','slandarer')
[nXr,nYr,nZr]=rotateXYZ(Xa, Ya, Za + .7,R_x_1);
nYr=nYr-.4;
surface(nXr,nYr,nZr-.1,'EdgeAlpha',0.05,...
'EdgeColor',[0 0 0],'FaceColor',[0.4 0.2 0])
drawStraw(nXr,nYr,nZr-.1)
for k=1:4
    [nXr,nYr,nZr]=rotateXYZ(nXr,nYr,nZr,R_z_2);
    surface(nXr,nYr,nZr-.1,'EdgeAlpha',0.05,...
    'EdgeColor',[0 0 0],'FaceColor',[0.4 0.2 0])
    drawStraw(nXr,nYr,nZr-.1)
end   
% -------------------------------------------------------------------------
[nXb,nYb,nZb]=rotateXYZ(xb./2.5,yb./2.5,Zb./2.5+.32,R_x_2);
nYb=nYb-1.35;
for k=1:5
    [nXb,nYb,nZb]=rotateXYZ(nXb,nYb,nZb,R_z_2);
    SHdl = surface(nXb,nYb,nZb,'EdgeColor','none','FaceColor','interp','CData',colorMapb);
    material(SHdl, 'dull')
    drawStraw(nXb,nYb,nZb)
end
[nXb,nYb,nZb]=rotateXYZ(xb./2.5,yb./2.5,Zb./2.5+.32,R_x_2);
nYb=nYb-1.15;
[nXb,nYb,nZb]=rotateXYZ(nXb,nYb,nZb,R_z_1);
for k=1:5
    [nXb,nYb,nZb]=rotateXYZ(nXb,nYb,nZb,R_z_2);
    SHdl = surface(nXb,nYb,nZb,'EdgeColor','none','FaceColor','interp','CData',colorMapb);
    material(SHdl, 'dull')
    drawStraw(nXb,nYb,nZb)
end
[nXb,nYb,nZb]=rotateXYZ(xb./2.5,yb./2.5,Zb./2.5+.32,R_x_2);
nYb=nYb-1.25;
[nXb,nYb,nZb]=rotateXYZ(nXb,nYb,nZb,R_z_3);
for k=1:5
    [nXb,nYb,nZb]=rotateXYZ(nXb,nYb,nZb,R_z_2);
    SHdl = surface(nXb,nYb,nZb,'EdgeColor','none','FaceColor','interp','CData',colorMapb);
    material(SHdl, 'dull')
    drawStraw(nXb,nYb,nZb)
end
[nXb,nYb,nZb]=rotateXYZ(xb./2.5,yb./2.5,Zb./2.5+.32,R_x_2);
nYb=nYb-1.25;
[nXb,nYb,nZb]=rotateXYZ(nXb,nYb,nZb,R_z_3);
[nXb,nYb,nZb]=rotateXYZ(nXb,nYb,nZb,R_z_3);
for k=1:5
    [nXb,nYb,nZb]=rotateXYZ(nXb,nYb,nZb,R_z_2);
    SHdl = surface(nXb,nYb,nZb,'EdgeColor','none','FaceColor','interp','CData',colorMapb);
    material(SHdl, 'dull')
    drawStraw(nXb,nYb,nZb)
end
%axes属性调整 ==============================================================
view(-15,35); axis equal off
camlight headlight; lighting gouraud;
 

function [X, Y, Z] = shitshape()
% 路径采样点数
points = 400;            
% 控制收缩曲线（后期快速变细）
z = linspace(0, 1, points);
z(1:50) = z(1:50) + z(50).*linspace(1, 0, 50);
z(201:400) = z(201:400) - .12.*linspace(0, 1, 200);
s = linspace(0, 1, points);
% 螺旋路径的半径变化（路径本身）
path_radius = linspace(.7, 0.02, points).*(cos(linspace(0, pi/2, points))+1.5)/3.5;
% 桶体粗细也收缩（同步缩小）
tube_radius = 0.18 * (1 - s.^2.5)+.001;
tube_radius(1:50) = sin(linspace(0, pi/2, 50)).*tube_radius(50);
% 螺旋角度
theta = linspace(0, 2*pi*4, points);
% 横截面圆点
circle_pts = 40;
circle_theta = linspace(0, 2*pi, circle_pts);
% 初始化
X = zeros(circle_pts, points);
Y = zeros(circle_pts, points);
Z = zeros(circle_pts, points);
for i = 1:(points - 1)
    % 当前螺旋中心位置（路径本身在缩小）
    R = path_radius(i);
    center_x = R * cos(theta(i));
    center_y = R * sin(theta(i));
    center_z = z(i);
    % 切向量（方向）
    dx = path_radius(i+1)*cos(theta(i+1)) - center_x;
    dy = path_radius(i+1)*sin(theta(i+1)) - center_y;
    dz = z(i+1) - center_z;
    tangent = [dx; dy; dz]; tangent = tangent / norm(tangent);
    % 横截面正交平面
    ref = [0; 0; 1];
    if abs(dot(ref, tangent)) > 0.99
        ref = [1; 0; 0];
    end
    normal1 = cross(tangent, ref); normal1 = normal1 / norm(normal1);
    normal2 = cross(tangent, normal1);
    % 横截面圆柱壳
    r_tube = tube_radius(i);
    for j = 1:circle_pts
        offset = r_tube * (cos(circle_theta(j)) * normal1 + sin(circle_theta(j)) * normal2);
        X(j,i) = center_x + offset(1);
        Y(j,i) = center_y + offset(2);
        Z(j,i) = center_z + offset(3);
    end
end
Z(:, end) = Z(:, end - 1);
end