% 
clc;close all;clear;
ephTable = readEphs;
Toc = arrayfun(@calSecFunc,ephTable.time);  % 计算参考时间
ephTable = addvars(ephTable,Toc,'before','PRN');  % 观测表
prn = unique(ephTable.PRN);
ecef= cell(numel(prn),1);
for ii = 1:numel(prn)
    eph = ephTable(ephTable.PRN == prn(ii),:);
    eph_interp = retime(eph,'minutely','previous'); % 差值  按照之前的值进行
    sec = arrayfun(@calSecFunc,eph_interp.time);  % 计算相应时间
    eph_interp = addvars(eph_interp,sec,'before','Toc');  % 加入参考时间秒在参考时间之前
    ecefPos  = rowfun(@satPos,eph_interp,'OutputVariableNames','pos');  %计算位置
    ecef{ii} = addvars(ecefPos,eph_interp.PRN,'NewVariableNames','PRN');  %在最后加入PRN
end
ecefTable = cat(1,ecef{:});  %将所有位置串联在一起
geoTable = calSatPos(ecefTable);
geoTable = sortrows(geoTable);
dupTimes = unique(geoTable.time);
[cdata,~,alpha] = imread('sat.png');

%%
figure('WindowState','maximized','MenuBar','none');
ax = axes(gcf,'Visible','off');
text(ax,0.5,0.55,'根据星历文件绘制北斗卫星动态运行轨迹',...
    'FontWeight','bold',...
    'FontName','宋体',...
    'FontSize',36,...
    'HorizontalAlignment','center');
A = print('-RGBImage','-r0');
%%
v = VideoWriter('output','MPEG-4');
open(v);
%% 
for ii = 1:60
    writeVideo(v,A);
end
%% 
ax1 = createWorldMap;
for ii = 1:63
    maph(ii) = addSatellites(ax1,0,0,cdata,alpha);
end
mt = textm(-100,-110,'yyyy-mm-dd HH:MM:SS',...
    'FontSize',32,'FontName',...
    'Times New Roman',...
    'FontWeight','Bold');
funPixelPos = @(lat,lon) calPixelPos(ax1.UserData,lat,lon,cdata);

for ii = 1:numel(dupTimes)
    t = dupTimes(ii);
    data = rowfun(funPixelPos,geoTable(t,1:2),'NumOutputs',2,'OutputFormat','cell');
    for jj = 1:63
        if jj<=size(data,1)
            maph(jj).Visible = 'on';
            maph(jj).XData = data{jj,1};
            maph(jj).YData = data{jj,2};
        else
            maph(jj).Visible = 'off';
        end
    end
    mt.String = datestr(t,'yyyy-mm-dd HH:MM:SS');
    drawnow
    AA = print(ax1.Parent,'-RGBImage','-r0');
    writeVideo(v,AA);
end
%% 
close(v)