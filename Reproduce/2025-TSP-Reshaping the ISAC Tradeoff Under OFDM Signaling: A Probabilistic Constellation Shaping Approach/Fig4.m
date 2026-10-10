%% Fig. 4：16-QAM 与 64-QAM 的启发式概率星座整形
clear;
clc;
close all;

width = 8;
height = 7;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

% 论文式(38)：使用凸二次规划求解启发式PCS，需要 Optimization Toolbox。
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off','OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10);

% 论文式(1)、图4：用 MATLAB 自带调制函数生成单位平均功率的完整QAM星座。
x16 = qammod((0:15).',16,'UnitAveragePower',true);
x64 = qammod((0:63).',64,'UnitAveragePower',true);

% 论文式(38)、(39)：以同心圆总概率为优化变量，再均分给该圆内的星座点。
p16Uniform = ones(16,1)/16;
p16PCS1 = solveHeuristicPCS(x16,1,options);
p16PCS115 = solveHeuristicPCS(x16,1.15,options);
p64Uniform = ones(64,1)/64;
p64PCS1 = solveHeuristicPCS(x64,1,options);
p64PCS125 = solveHeuristicPCS(x64,1.25,options);

constellations = {x16,x16,x16,x64,x64,x64};
probabilities = {p16Uniform,p16PCS1,p16PCS115,p64Uniform,p64PCS1,p64PCS125};
titles = {'(a) 16-QAM','(b) 16-QAM-PCS: $c_0=1$','(c) 16-QAM-PCS: $c_0=1.15$','(d) 64-QAM','(e) 64-QAM-PCS: $c_0=1$','(f) 64-QAM-PCS: $c_0=1.25$'};

% 论文式(38)：逐图检查概率归一化、平均功率和实际达到的四阶矩。
fprintf('子图    概率之和       平均功率       四阶矩\n');
for ipanel = 1:6
    x = constellations{ipanel};
    p = probabilities{ipanel};
    powerMoment = sum(p.*abs(x).^2);
    fourthMoment = sum(p.*abs(x).^4);
    fprintf('(%c)     %.8f     %.8f     %.8f\n',char('a'+ipanel-1),sum(p),powerMoment,fourthMoment);
    assert(min(p)>-1e-8 && abs(sum(p)-1)<1e-6 && abs(powerMoment-1)<1e-6,'概率或平均功率约束核对未通过。');
end
assert(abs(sum(p16PCS1.*abs(x16).^4)-1)<1e-5,'16-QAM的最小四阶矩核对未通过。');
assert(abs(sum(p64PCS1.*abs(x64).^4)-457/441)<1e-5,'64-QAM的最小四阶矩核对未通过。');
assert(abs(sum(p16PCS115.*abs(x16).^4)-1.15)<1e-5,'16-QAM在c0=1.15时的目标四阶矩核对未通过。');
assert(abs(sum(p64PCS125.*abs(x64).^4)-1.25)<1e-5,'64-QAM在c0=1.25时的目标四阶矩核对未通过。');

% 论文图4：两排三个概率星座图；零概率背景使用parula的最低颜色。
figure(4);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
colormap(gcf,parula(256));
leftPosition = [0.045,0.365,0.685];
bottomPosition = [0.59,0.11];
axesPosition = zeros(6,4);
axesHandles = gobjects(6,1);

for ipanel = 1:6
    x = constellations{ipanel};
    p = probabilities{ipanel};
    if ipanel <= 3
        coordinateAxis = linspace(-4,4,161)/sqrt(10);
        row = 1;
    else
        coordinateAxis = linspace(-8,8,321)/sqrt(42);
        row = 2;
    end
    probabilityMap = zeros(numel(coordinateAxis));
    for iq = 1:numel(x)
        [~,ix] = min(abs(coordinateAxis-real(x(iq))));
        [~,iy] = min(abs(coordinateAxis-imag(x(iq))));
        probabilityMap(iy-1:iy+1,ix-1:ix+1) = p(iq);
    end
    column = mod(ipanel-1,3)+1;
    axesPosition(ipanel,:) = [leftPosition(column),bottomPosition(row),0.27,0.36];
    axesHandles(ipanel) = axes('Parent',gcf,'Units','normalized','Position',axesPosition(ipanel,:));
    imagesc(axesHandles(ipanel),coordinateAxis,coordinateAxis,probabilityMap);
    set(axesHandles(ipanel),'YDir','normal');
    axis(axesHandles(ipanel),'image');
    axis(axesHandles(ipanel),'off');
    % 原图各面板独立缩放颜色；每行色条绑定中间面板(b)或(e)。
    caxis(axesHandles(ipanel),[0,max(p)]);
    title(axesHandles(ipanel),titles{ipanel},'Interpreter','latex','FontName','Times New Roman','FontSize',fontsize,'FontWeight','normal');
end

colorbar16 = colorbar(axesHandles(2),'Location','southoutside');
set(colorbar16,'Units','normalized','Position',[0.14,0.535,0.72,0.02],'FontName','Times New Roman','FontSize',12,'Ticks',0:0.02:0.12);
colorbar64 = colorbar(axesHandles(5),'Location','southoutside');
set(colorbar64,'Units','normalized','Position',[0.14,0.055,0.72,0.02],'FontName','Times New Roman','FontSize',12,'Ticks',0:0.01:0.06);
for ipanel = 1:6
    set(axesHandles(ipanel),'Position',axesPosition(ipanel,:));
end
drawnow;
% print(gcf,'Fig4_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig4_2024_TSP.pdf','-dpdf','-vector');

function p = solveHeuristicPCS(x,c0,options)
    % 论文式(39)：将同半径星座点分组；q为圆环总概率，p为单个星座点概率。
    [ringPower,~,ringIndex] = unique(round(abs(x).^2,12));
    numberOfRings = numel(ringPower);
    ringCount = accumarray(ringIndex,1,[numberOfRings,1]);
    fourthPower = ringPower.^2;
    Aeq = [ones(1,numberOfRings);ringPower.'];
    beq = [1;1];
    if c0 == 1
        % 论文式(38)：四阶矩不小于1，最优解只使用功率1两侧相邻的圆环。
        lowerRing = find(ringPower<=1,1,'last');
        upperRing = find(ringPower>=1,1,'first');
        ringMass = zeros(numberOfRings,1);
        if lowerRing == upperRing
            ringMass(lowerRing) = 1;
        else
            ringMass(lowerRing) = (ringPower(upperRing)-1)/(ringPower(upperRing)-ringPower(lowerRing));
            ringMass(upperRing) = 1-ringMass(lowerRing);
        end
    else
        % 论文式(38)：最小化(fourthPower'*q-c0)^2，省略与q无关的常数c0^2。
        H = 2*(fourthPower*fourthPower.');
        f = -2*c0*fourthPower;
        initialMass = ringCount/numel(x);
        [ringMass,~,exitflag] = quadprog(H,f,[],[],Aeq,beq,zeros(numberOfRings,1),ones(numberOfRings,1),initialMass,options);
        if exitflag <= 0
            error('式(38)的二次规划求解失败，exitflag=%d。',exitflag);
        end
    end
    p = ringMass(ringIndex)./ringCount(ringIndex);
end