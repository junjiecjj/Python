function plotJRCScenario(jrcPosition, targetPositions, userPosition, targetVelocities)
    figure;
    hold on;
    plot(jrcPosition(2), jrcPosition(1), '^', 'Color', 'g', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'JRC');
    plot(targetPositions(2, :), targetPositions(1, :), 's', 'Color', 'b', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', 'Radar targets');
    
    if ~isempty(userPosition)
        plot(userPosition(2), userPosition(1), 'p', 'Color', 'r', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Downlink user');
    end
    
    quiver(targetPositions(2, :), targetPositions(1, :), targetVelocities(2, :), targetVelocities(1, :), 'HandleVisibility', 'off');
    
    for i = 1:size(targetPositions, 2)
        targetSpeed = norm(targetVelocities(:, i));
        labelX = targetPositions(2, i) + 0.3*targetVelocities(2, i) + 0.5;
        labelY = targetPositions(1, i) + 0.3*targetVelocities(1, i) + 0.5;
        text(labelX, labelY, sprintf('%.1f m/s', targetSpeed));
    end   
    
    grid on;
    xlabel('y (m)');
    ylabel('x (m)');
    legend('Location', 'southoutside', 'Orientation', 'horizontal');
    xlim([-25 25]);
    title('JRC Scenario');
end