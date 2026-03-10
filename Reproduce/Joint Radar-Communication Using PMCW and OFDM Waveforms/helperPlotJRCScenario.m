function helperPlotJRCScenario(jrcpos, tgtpos, userpos, tgtvel)
    figure;
    hold on;
    plot(jrcpos(2), jrcpos(1), '^', 'Color', 'g', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'JRC');
    plot(tgtpos(2, :), tgtpos(1, :), 's', 'Color', 'b', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', 'Radar targets');
    
    if ~isempty(userpos)
        plot(userpos(2), userpos(1), 'p', 'Color', 'r', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Downlink user');
    end
    
    quiver(tgtpos(2, :), tgtpos(1, :), tgtvel(2, :), tgtvel(1, :), 'HandleVisibility', 'off');
    
    for i = 1:size(tgtpos, 2)
        s = norm(tgtvel(:, i));
        x = tgtpos(2, i) + 0.3*tgtvel(2, i) + 0.5;
        y = tgtpos(1, i) + 0.3*tgtvel(1, i) + 0.5;
        text(x, y, sprintf('%.1f m/s', s));
    end   
    
    grid on;
    xlabel('y (m)');
    ylabel('x (m)');
    legend('Location', 'southoutside', 'Orientation', 'horizontal');
    xlim([-25 25]);
    title('JRC Scenario');
end