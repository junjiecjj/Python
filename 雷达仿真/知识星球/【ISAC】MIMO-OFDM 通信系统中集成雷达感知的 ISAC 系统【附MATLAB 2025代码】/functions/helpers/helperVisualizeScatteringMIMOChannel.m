function helperVisualizeScatteringMIMOChannel(scatteringMIMOChannel, scattererPositions, targetPositions, targetVelocities)
    figure;
    hold on;

    ax = gca;
    colors = ax.ColorOrder;

    xspan = [min(scattererPositions(1, :)) max(scattererPositions(1, :))];
    dx = (xspan(2) - xspan(1))/10;

    yspan = [min(scattererPositions(2, :)) max(scattererPositions(2, :))];
    dy = (yspan(2) - yspan(1))/10;
    
    txPosition = scatteringMIMOChannel.TransmitArrayPosition;
    txOrientationAxis = scatteringMIMOChannel.TransmitArrayOrientationAxes;
    plot(txPosition(2), txPosition(1), 'o', 'Color', colors(5, :), 'MarkerSize', 12, 'MarkerFaceColor', colors(5, :), 'LineWidth', 1.5, 'DisplayName', 'Tx');
    quiver(txPosition(2), txPosition(1), dx*txOrientationAxis(2,1), dx*txOrientationAxis(1, 1),...
        'Color', colors(5, :), 'LineWidth', 1.5, 'DisplayName', 'Tx array normal', 'MaxHeadSize', 1);

    rxPosition = scatteringMIMOChannel.ReceiveArrayPosition;
    rxOrientationAxis = scatteringMIMOChannel.ReceiveArrayOrientationAxes;    
    plot(rxPosition(2), rxPosition(1), 'v', 'Color', colors(3, :), 'MarkerSize', 10, 'MarkerFaceColor', colors(3, :), 'LineWidth', 1.5, 'DisplayName', 'Rx');
    quiver(rxPosition(2), rxPosition(1), dy*rxOrientationAxis(2,1), dy*rxOrientationAxis(1, 1),...
        'Color', colors(3, :), 'LineWidth', 1.5, 'DisplayName', 'Rx array normal', 'MaxHeadSize', 1);
    
    plot(scattererPositions(2, :), scattererPositions(1, :), '.', 'Color', colors(1, :), 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Static scatterers');   
    plot(targetPositions(2, :), targetPositions(1, :), 'x', 'Color', colors(2, :), 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Moving targets');
        
    for i = 1:size(targetPositions, 2)
        quiver(targetPositions(2, i), targetPositions(1, i), targetVelocities(2, i), targetVelocities(1, i),...
            'HandleVisibility', 'off', 'Color', colors(2, :), 'LineWidth', 1.5, 'MaxHeadSize', 1);
        s = norm(targetVelocities(:, i));
        x = targetPositions(2, i) + 0.3*targetVelocities(2, i) + 2;
        y = targetPositions(1, i) + 0.3*targetVelocities(1, i) + 2;
        text(x, y, sprintf('%.1f m/s', s));
    end   
    
    grid on;
    xlabel('y (m)');
    ylabel('x (m)');
    legend('Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 2);
    % xlim([-25 25]);
    axis equal;
    box on;
end