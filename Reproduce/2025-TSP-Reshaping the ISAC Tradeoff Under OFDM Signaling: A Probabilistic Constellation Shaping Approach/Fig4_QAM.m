%% Complete Fig. 4 -- Du et al., IEEE TSP (2024)
% Six panels and both colorbars, with the palette extracted from the paper.
% Panels (a)-(e): analytical probabilities from P3, Eq. (38).
% Panel (f): the published solution is nonunique. Its color indices are
% digitized from the original PDF and projected onto the exact moment
% constraints. This recovers its displayed colors, not undisclosed raw data.
% Requires Communications Toolbox (qammod). Local functions are at the end.

clear; clc; close all;

% 文章 Assumption 1（式(9)前）：星座满足零均值、单位功率和旋转对称。
% P1 的 C2：sum_x p(x)*A_x^2 = 1，其中 A_x = abs(x)。
% 此处生成均匀基准星座，UnitAveragePower=true 实现 (1/Q)*sum_x abs(x)^2 = 1。
% Generate the QAM constellations with MATLAB's modulation function.
x16 = qammod((0:15).',16,'gray','UnitAveragePower',true);
x64 = qammod((0:63).',64,'gray','UnitAveragePower',true);

% 对应文章 Assumption 1：同一同心圆上的星座点具有相同概率。
% 为实现式(39)的降维，定义 e_j=A_j^2、n_j=第j环的点数。
% 后续 w_j 为第j环总概率；该环每个点的概率为 q_j=w_j/n_j。
% Group the built-in constellation points by squared radius.
% Rounding is used only for grouping, not for changing constellation points.
[~,~,id16] = unique(round(abs(x16).^2,12));
e16 = accumarray(id16,abs(x16).^2,[],@mean);
n16 = accumarray(id16,1);
[~,~,id64] = unique(round(abs(x64).^2,12));
e64 = accumarray(id64,abs(x64).^2,[],@mean);
n64 = accumarray(id64,1);
p16 = zeros(16,3); p64 = zeros(64,3);
% Fig.4(a)、(d)的均匀QAM基准：p(x)=1/Q，满足P1的C3：sum_x p(x)=1。
p16(:,1) = 1/16;
p64(:,1) = 1/64;

% 16-QAM has three rings. Ring masses include all points on each ring.
% 文章式(38)：min_p (sum_x p(x)*A_x^4-c0)^2，s.t. C2、C3、C4。
% 16-QAM有3个环；目标四阶矩可达时，式(38)的最小目标值为0。
% 由P1的C1-C3及式(39)，使用环总概率w_j，并将矩阵行顺序重排：
%   [1 ... 1; e_1 ... e_W; e_1^2 ... e_W^2]*w = [1;1;c0], w_j>=0。
% 即 sum_j w_j=1、sum_j e_j*w_j=1、sum_j e_j^2*w_j=c0。
A16 = [ones(1,3); e16.'; (e16.^2).'];
w16_1 = A16\[1;1;1];
w16_115 = A16\[1;1;1.15];
% 实现换元（由同环等概率及概率归一化得到）：p(x)=w_j/n_j，x属于第j环。
p16(:,2) = w16_1(id16)./n16(id16);
p16(:,3) = w16_115(id16)./n16(id16);
p16(abs(p16)<1e-13) = 0;

% 64-QAM cannot achieve c0=1 with unit mean power. P3 chooses the
% two rings surrounding energy 1; the resulting fourth moment is 457/441.
% 对应式(38)的P3；64-QAM不存在e_j=1的环，故c0=1不可达。
% 以下为补充推导，论文没有给出该闭式解的独立公式编号：
% 令a=max{e_j:e_j<1}、b=min{e_j:e_j>1}，在这两个相邻环上分配质量。
% 从C2、C3解得：w_a+w_b=1，a*w_a+b*w_b=1，
% 因而 w_a=(b-1)/(b-a)，w_b=(1-a)/(b-a)。
% 最小四阶矩 m4_min=a^2*w_a+b^2*w_b=457/441，对应Fig.4(e)的1.0363。
lo = find(e64<1,1,'last');
hi = find(e64>1,1,'first');
w64_1 = zeros(numel(e64),1);
w64_1(lo) = (e64(hi)-1)/(e64(hi)-e64(lo));
w64_1(hi) = 1-w64_1(lo);
% 同环等概率的实现换元：p(x)=w_j/n_j；上述w_j满足P3的C2-C4。
p64(:,2) = w64_1(id64)./n64(id64);

% Fig. 4(f): zero-based color indices read from the embedded PDF image.
% Rings are ordered by squared radius [2 10 18 26 34 50 58 74 98]/42.
% The original plot scales each panel by its own maximum probability.
% 以下颜色索引及初始概率来自Fig.4(f)的数字化，不是文章给出的求解公式。
% 式(39)下64-QAM有9个环和3个等式，论文说明其解可能不唯一。
sourceIndex64 = [96;118;148;191;241;255;206;126;74];
ratio = (sourceIndex64+0.5)/256;
ratio(sourceIndex64==255) = 1;
qInitial = ratio/sum(n64.*ratio);
% P1的C1-C3 / 式(39)对应的逐点概率写法，显式计入每环的点数n_j：
%   sum_j n_j*q_j=1，sum_j n_j*e_j*q_j=1，sum_j n_j*e_j^2*q_j=c0。
% 这里q_j指第j环每个点的概率，A64*q=[1;1;1.25]。
A64 = [n64.'; (n64.*e64).'; (n64.*e64.^2).'];
% Orthogonal correction removes digitization error in the three moments.
% 数字化重建所用的补充公式，非原文公式：
%   q=q0+A64.'*(A64*A64.')^(-1)*(b0-A64*q0)，b0=[1;1;1.25]。
% 该投影将q0校正到上面的矩约束平面；随后检查非负性及原图颜色索引。
% 满足四阶矩1.25后，式(38)的P3目标值为0。
q64_125 = qInitial + A64.'*((A64*A64.')\([1;1;1.25]-A64*qInitial));
p64(:,3) = q64_125(id64);
assert(isequal(colorIndex(q64_125),sourceIndex64), 'The reconstructed probabilities must retain the original colors.');

% Numerical checks of nonnegativity, normalization, and average power.
checkProbabilities(x16,p16,[1.32,1,1.15]);
checkProbabilities(x64,p64,[29/21,457/441,1.25]);
% 计算P1的C1及式(38)中出现的四阶矩：m4=E{A_x^4}=sum_x p(x)*abs(x)^4。
fprintf('16-QAM fourth moments: %.9f  %.9f  %.9f\n', sum(p16.*abs(x16).^4,1));
fprintf('64-QAM fourth moments: %.9f  %.9f  %.9f\n', sum(p64.*abs(x64).^4,1));

% 使用MATLAB内置parula配色，接近原文Fig.4。
colorMap = parula(256);
pointMap = colorMap;
backgroundColor = colorMap(1,:);

% The PDF stores point colors with slightly different RGB rounding.
pointMap = colorMap;
pointIndex = [74 78 96 118 126 148 170 191 206 241 255];
pointMap(pointIndex+1,:) = [ ...
    45 136 245; ...
    44 141 241; ...
    32 163 227; ...
    0 184 200; ...
    12 189 187; ...
    59 201 145; ...
    128 203 88; ...
    199 193 41; ...
    238 185 52; ...
    244 230 37; ...
    249 250 20]/255;
backgroundColor = [61,38,168]/255;

fig = figure(1);
clf(fig);
set(fig,'Color','white','Units','inches', 'Position',[1,1,11,9.4],'Renderer','painters');
set(fig,'PaperUnits','inches','PaperPosition',[0,0,11,9.4], 'PaperSize',[11,9.4]);
colormap(fig,colorMap);
panelX = [0.055,0.3725,0.69];
panelY = [0.59,0.145];
panelW = 0.255; panelH = 0.34;

% Calculate the kurtosis of the uniform QAM distributions in column 1.
% 归一化四阶矩定义（论文未单独编号）：kurtosis=E{|x|^4}/(E{|x|^2})^2。
% 根据P1的C2，E{|x|^2}=1，因此kurtosis等于式(38)中的实际四阶矩。
kurtosis16 = sum(p16(:,1).*abs(x16).^4) /sum(p16(:,1).*abs(x16).^2)^2;
kurtosis64 = sum(p64(:,1).*abs(x64).^4) /sum(p64(:,1).*abs(x64).^2)^2;

titles = { ...
    sprintf('(a) 16-QAM, kurtosis = %.2f',kurtosis16), '(b) 16-QAM-PCS: kurtosis=1', '(c) 16-QAM-PCS: kurtosis=1.15'; ...
    sprintf('(d) 64-QAM, kurtosis = %.2f',kurtosis64), '(e) 64-QAM-PCS: kurtosis=1', '(f) 64-QAM-PCS: kurtosis=1.25'};

for row = 1:2
    if row==1
        x = x16; probabilities = p16; lim = 5/sqrt(10); pointSize = 16;
    else
        x = x64; probabilities = p64; lim = 9/sqrt(42); pointSize = 10;
    end
    for col = 1:3
        ax = axes(fig,'Position',[panelX(col),panelY(row),panelW,panelH], 'Color',backgroundColor);
        hold(ax,'on');
        pj = probabilities(:,col);
        visible = pj>1e-12;
        rgb = pointMap(colorIndex(pj)+1,:);
        scatter(ax,real(x(visible)),imag(x(visible)), pointSize,rgb(visible,:),'s','filled');
        set(ax,'XLim',[-lim,lim],'YLim',[-lim,lim], 'XTick',[],'YTick',[],'XColor','none','YColor','none', 'Box','off','FontName','Arial','FontSize',13);
        title(ax,titles{row,col},'Interpreter','tex', 'FontName','Arial','FontWeight','normal','FontSize',13);
    end
end

% The paper uses per-panel color normalization, but labels each row's
% shared bar using the maximum probability in its c0=1 panel.
% Do not apply a common caxis to all six panels: that changes their colors.
makeColorbar(fig,[0.135,0.542,0.73,0.014],1/8,0:0.02:0.12);
makeColorbar(fig,[0.135,0.097,0.73,0.014],1/16,0:0.01:0.06);
drawnow;

outputDir = fileparts(mfilename('fullpath'));
if isempty(outputDir), outputDir = pwd; end
print(fig,fullfile(outputDir,'Fig4_full.png'),'-dpng','-r600');
print(fig,fullfile(outputDir,'Fig4_full.pdf'),'-dpdf','-painters');
% save(fullfile(outputDir,'Fig4_full_data.mat'), 'x16','x64','p16','p64','e16','e64','n16','n64', 'q64_125','sourceIndex64','colorMap','pointMap','backgroundColor');

%% 绘图：仅保留三个64-QAM子图，颜色条放在上方
width = 12;
height = 5. ;
fontsize = 18;
fig = figure(2);
clf(fig);
set(fig,'Units','inches','Position',[1,1,width,height],'Color','white','Renderer','painters');
set(fig,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
colormap(fig,colorMap);

panelPositions = [0.055,0.055,0.25,0.665; 0.385,0.055,0.25,0.665; 0.715,0.055,0.25,0.665];
panelTitles = {'64-QAM, kurtosis = 1.38','64-QAM-PCS, kurtosis = 1','64-QAM-PCS, kurtosis = 1.25'};
coordinateLimit = 9/sqrt(42);
backgroundRGB = repmat(reshape(backgroundColor,1,1,3),2,2,1);

for col = 1:3
    ax = axes(fig,'Units','normalized','Position',panelPositions(col,:));
    hold(ax,'on');
    % 使用真实RGB图像绘制底色，不依赖axes的Color属性。
    image(ax,'XData',[-coordinateLimit,coordinateLimit],'YData',[-coordinateLimit,coordinateLimit],'CData',backgroundRGB);
    pj = p64(:,col);
    visible = pj>1e-12;
    index = min(255,max(0,floor(256*pj/max(pj))));
    rgb = pointMap(index+1,:);
    scatter(ax,real(x64(visible)),imag(x64(visible)),10,rgb(visible,:),'s','filled');
    set(ax,'XLim',[-coordinateLimit,coordinateLimit],'YLim',[-coordinateLimit,coordinateLimit],'YDir','normal','XTick',[],'YTick',[],'Box','off','FontName','Times New Roman','FontSize',fontsize);
    % 关闭坐标轴后，再明确显示标题。
    set(ax,'Visible','off');
    hTitle = title(ax,panelTitles{col},'Interpreter','tex','FontName','Times New Roman','FontSize',fontsize,'FontWeight','normal');
    set(hTitle,'Visible','on','Color','black');
end

%% 上方颜色条：刻度在上侧，Probability写在条内
colorLimits = [0,1/16];
barAxes = axes(fig,'Units','normalized','Position',[0.055,0.81,0.91,0.09]);
barValues = linspace(colorLimits(1),colorLimits(2),256);
imagesc(barAxes,barValues,[0,1],repmat(barValues,2,1));
colormap(barAxes,colorMap);
caxis(barAxes,colorLimits);
set(barAxes,'XLim',colorLimits,'YLim',[0,1],'YDir','normal','YTick',[],'XAxisLocation','top','XTick',0:0.01:0.06,'XTickLabel',{'0','0.01','0.02','0.03','0.04','0.05','0.06'},'TickDir','in','Box','on','Layer','top','LineWidth',1,'FontName','Times New Roman','FontSize',fontsize);
text(barAxes,0.97,0.45,'Probability','Units','normalized','HorizontalAlignment','right','VerticalAlignment','middle','FontName','Times New Roman','FontSize',fontsize,'Interpreter','none','Color','black');

%% 导出
drawnow;
print(fig,'Fig4_64QAM.png','-dpng','-r600');
print(fig,'Fig4_64QAM.pdf','-dpdf','-painters');

% 绘图辅助函数，无原文公式编号；实现各子图按自身最大概率归一化的颜色映射。
function k = colorIndex(p)
    % MATLAB's scaled lookup with 256 colors, starting at probability zero.
    k = min(255,max(0,floor(256*p/max(p))));
end

function checkProbabilities(x,p,targetFourth)
    % P1的C4 / 式(38)约束：p(x)>=0；容差用于允许浮点舍入误差。
    assert(min(p(:))>=-1e-12,'Negative probability.');
    % P1的C3 / 式(38)约束：sum_x p(x)=1。
    assert(max(abs(sum(p,1)-1))<1e-10,'Probabilities do not sum to 1.');
    % P1的C2 / 式(38)约束：sum_x p(x)*A_x^2=1。
    assert(max(abs(sum(p.*abs(x).^2,1)-1))<1e-10,'Mean power is not 1.');
    % 检验式(38)目标中m4=sum_x p(x)*A_x^4的实际值。
    % targetFourth为均匀基准、可达目标或c0不可达时的最小可达四阶矩。
    assert(max(abs(sum(p.*abs(x).^4,1)-targetFourth))<1e-10, 'Unexpected fourth moment.');
end

% 绘图辅助函数：按照Fig.4布局绘制颜色条，无原文公式对应。
function makeColorbar(fig,position,maxValue,ticks)
    ax = axes(fig,'Position',position);
    caxis(ax,[0,maxValue]);
    cb = colorbar(ax,'southoutside');
    cb.Position = position;
    cb.Ticks = ticks;
    cb.TickLabels = arrayfun(@(t)sprintf('%g',t),ticks,'UniformOutput',false);
    cb.FontName = 'Arial'; cb.FontSize = 12;
    cb.TickDirection = 'in'; cb.LineWidth = 0.8;
    ax.Visible = 'off';
end