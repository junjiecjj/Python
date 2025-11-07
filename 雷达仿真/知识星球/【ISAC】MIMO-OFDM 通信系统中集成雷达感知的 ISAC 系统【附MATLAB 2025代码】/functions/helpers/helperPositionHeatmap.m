classdef helperPositionHeatmap < handle
    properties
        ReceiveArray
        ReceiveArrayOrientationAxis
        SampleRate
        CarrierFrequency
        Bandwidth
        OFDMSymbolDuration
        TransmitArrayPosition
        ReceiveArrayPosition
        TargetPositions
        ROI
        TransmitArrayOrientationAxis
    end
        
    methods
        function obj = helperPositionHeatmap(varargin)
            for i = 1:2:(nargin-1)
                obj.(varargin{i}) = varargin{i+1};
            end
        end

        function plot(obj, data)
            x = obj.ROI(1, 1) : 0.5 : obj.ROI(1, 2);
            y = obj.ROI(2, 1) : 0.5 : obj.ROI(2, 2);
            [X, Y] = meshgrid(x, y);
            S = [X(:) Y(:) zeros(numel(X), 1)].';
        
            rtx = sqrt(sum((obj.TransmitArrayPosition - S).^2));
            rrx = sqrt(sum((obj.ReceiveArrayPosition - S).^2));
        
            rq = rtx + rrx;
            arrayAxis = obj.ReceiveArrayOrientationAxis*[0; 1; 0];
            thetaq = 90 - acosd(arrayAxis'*(S - obj.ReceiveArrayPosition)./rrx);
        
            rangeAngleResponse = phased.RangeAngleResponse('SensorArray', obj.ReceiveArray, 'RangeMethod', 'FFT', ...
                'SampleRate', obj.SampleRate, 'SweepSlope', obj.Bandwidth/obj.OFDMSymbolDuration,...
                'OperatingFrequency', obj.CarrierFrequency, 'ReferenceRangeCentered', false);
            
            [rar, r, theta] = rangeAngleResponse(conj(data));
            rar_abs = sum(abs(rar), 3);
            theta = theta * (-1);   % -1 to account for conj
            r = r*2;                % x2 since it is a bistaic range
            
            [thetav, Rv] = meshgrid(theta, r);
        
            xyr_abs = interp2(thetav, Rv, rar_abs, thetaq, rq);
            xyr_abs = xyr_abs./max(xyr_abs, [], 'all');
        
            imagesc(y, x, reshape(xyr_abs, numel(y), numel(x)).');
            ax = gca;
            set(ax, 'YDir', 'normal', 'GridColor', [1 1 1], 'GridLineWidth', 1) ;
            % colormap(ax, sky);
            hold on;
            
            colorbar;
            xlabel('X (m)');
            ylabel('Y (m)');
            title('Position Heatmap');
            grid on;

            dx = (obj.ROI(1, 2) - obj.ROI(1, 1))/10;
            dy = (obj.ROI(2, 2) - obj.ROI(2, 1))/10;

            colors = ax.ColorOrder;

            plot(obj.TransmitArrayPosition(2), obj.TransmitArrayPosition(1), 'o', 'Color', colors(5, :), 'MarkerSize', 12, 'MarkerFaceColor', colors(5, :), 'LineWidth', 1.5, 'DisplayName', 'Tx');
            quiver(obj.TransmitArrayPosition(2), obj.TransmitArrayPosition(1), dx*obj.TransmitArrayOrientationAxis(2, 1), dx*obj.TransmitArrayOrientationAxis(1, 1),...
                'Color', colors(5, :), 'LineWidth', 1.5, 'DisplayName', 'Tx array normal', 'MaxHeadSize', 1);
            
            plot(obj.ReceiveArrayPosition(2), obj.ReceiveArrayPosition(1), 'v', 'Color', colors(3, :), 'MarkerSize', 10, 'MarkerFaceColor', colors(3, :), 'LineWidth', 1.5, 'DisplayName', 'Rx');
            quiver(obj.ReceiveArrayPosition(2), obj.ReceiveArrayPosition(1), dy*obj.ReceiveArrayOrientationAxis(2, 1), dy*obj.ReceiveArrayOrientationAxis(1, 1),...
                'Color', colors(3, :), 'LineWidth', 1.5, 'DisplayName', 'Rx array normal', 'MaxHeadSize', 1);            

            plot(obj.TargetPositions(2, :), obj.TargetPositions(1, :), ...
                'x', 'Color', colors(2, :), 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Targets of interest');       
            legend('Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 2);
        end
    end
end