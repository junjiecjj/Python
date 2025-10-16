clc; clear
figure('Position',[300,50,900,900], 'Color','k');
axes(gcf, 'NextPlot','add', 'Position',[0,0,1,1], 'Color','k');
axis([0, 400, 0, 400])
SHdl = scatter([], [], 2, 'filled','o','w', 'MarkerEdgeColor','none', 'MarkerFaceAlpha',.4);
t = 0;
i = 1:1e4;
y = i./345;
x = y; idx = y < 11;
x(idx) = 6 + sin(bitxor(floor(x(idx)), 8))*6;
x(~idx) = x(~idx)./5 + cos(x(~idx)./2);
e = y./7 - 13;

while true
    t = t + pi/120;
    k = x.*cos(i - t./4);
    d = sqrt(k.^2 + e.^2) + sin(e./4 + t)./2;
    q = y.*k./d.*(3 + sin(d.*2 + y./2 - t.*4));
    c = d./2 + 1 - t./2;
    SHdl.XData = q + 60.*cos(c) + 200;
    SHdl.YData = 400 - (q.*sin(c) + d.*29 - 170);
    drawnow; pause(5e-3)
end