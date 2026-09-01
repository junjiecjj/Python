clear
close all
clc

%% ============================================================
%  12x16 configuration (12 TX x 16 RX)
% ============================================================

rxX_12x16 = [0:0.5:1.5 5.5:0.5:7 23:0.5:26.5];
rxY_12x16 = zeros(size(rxX_12x16));

txX_12x16 = [0 2 4 6 8 10 12 14 16 4.5 5 5.5];
txY_12x16 = [4 4 4 4 4 4 4 4 4 3.5 2 1];

rxPos_12x16 = [rxX_12x16(:), rxY_12x16(:)];
txPos_12x16 = [txX_12x16(:), txY_12x16(:)];

vaPos_12x16 = getVirtualArray(txPos_12x16, rxPos_12x16);

figure
subplot(1,2,1)
plotPhysicalArray(txPos_12x16, rxPos_12x16, '12x16: TX/RX Positions')

subplot(1,2,2)
plotVirtualArray(vaPos_12x16, '12x16: Virtual Array')


%% ============================================================
%  48x48 configuration (48 TX x 48 RX)
% ============================================================

rxBotX = 0:23;
rxTopX = 3.5:26.5;

rxBotY = 2 * ones(1,24) - 2;
rxTopY = 14 * ones(1,24) - 2;

txLeft1X  = 0.5 * ones(1,12);
txLeft2X  = 2.0 * ones(1,12);
txRight1X = 24.5 * ones(1,12);
txRight2X = 26.0 * ones(1,12);

txRight1Y = (2:13) - 2;
txRight2Y = (2:13) - 2;

txLeft1Y = (3:14) - 2;
txLeft2Y = (3:14) - 2;

rxPos_48x48 = [rxTopX, rxBotX; ...
               rxTopY, rxBotY].';

txPos_48x48 = [txLeft1X txLeft2X txRight1X txRight2X; ...
               txLeft1Y txLeft2Y txRight1Y txRight2Y].';

vaPos_48x48 = getVirtualArray(txPos_48x48, rxPos_48x48);

figure
subplot(1,2,1)
plotPhysicalArray(txPos_48x48, rxPos_48x48, '48x48: TX/RX Positions')

subplot(1,2,2)
plotVirtualArray(vaPos_48x48, '48x48: Virtual Array')


%% ============================================================
%  16x16_1 configuration (16 TX x 16 RX)
% ============================================================

txX_16x16_1 = [2 4 6 8 10 12 14 16 16.5 18.5 20.5 22.5 24.5 26.5 28.5 30.5];
txY_16x16_1 = [1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 0 0 0 0 0 0 0 0];

rxX_16x16_1 = [0 2.5 5 7.5 9 10.5 12 14.5 17 19.5 22 24.5 25 27.5 30 32.5];
rxY_16x16_1 = [13 13 13 13 13.5 13 13.5 13.5 13.5 13.5 13.5 13.5 14 14 14 14];

rxPos_16x16_1 = [rxX_16x16_1(:), rxY_16x16_1(:)];
txPos_16x16_1 = [txX_16x16_1(:), txY_16x16_1(:)];

vaPos_16x16_1 = getVirtualArray(txPos_16x16_1, rxPos_16x16_1);

figure
subplot(1,2,1)
plotPhysicalArray(txPos_16x16_1, rxPos_16x16_1, '16x16\_1: TX/RX Positions')

subplot(1,2,2)
plotVirtualArray(vaPos_16x16_1, '16x16\_1: Virtual Array')


%% ============================================================
%  16x16_2 configuration (16 TX x 16 RX)
% ============================================================

txX_16x16_2 = zeros(1,16);
txY_16x16_2 = 0:0.5:7.5;

rxX_16x16_2 = 3:0.5:10.5;
rxY_16x16_2 = [4 3.5 4 3.5 4 3.5 4 3.5 4 3.5 4 3.5 4 3.5 4 3.5];

rxPos_16x16_2 = [rxX_16x16_2(:), rxY_16x16_2(:)];
txPos_16x16_2 = [txX_16x16_2(:), txY_16x16_2(:)];

vaPos_16x16_2 = getVirtualArray(txPos_16x16_2, rxPos_16x16_2);

figure
subplot(1,2,1)
plotPhysicalArray(txPos_16x16_2, rxPos_16x16_2, '16x16\_2: TX/RX Positions')

subplot(1,2,2)
plotVirtualArray(vaPos_16x16_2, '16x16\_2: Virtual Array')


%% ============================================================
%  Print number of elements
% ============================================================

fprintf('\n12x16:\n')
fprintf('RX elements        = %d\n', size(rxPos_12x16,1))
fprintf('TX elements        = %d\n', size(txPos_12x16,1))
fprintf('Unique VA elements = %d\n', size(vaPos_12x16,1))

fprintf('\n48x48:\n')
fprintf('RX elements        = %d\n', size(rxPos_48x48,1))
fprintf('TX elements        = %d\n', size(txPos_48x48,1))
fprintf('Unique VA elements = %d\n', size(vaPos_48x48,1))

fprintf('\n16x16_1:\n')
fprintf('RX elements        = %d\n', size(rxPos_16x16_1,1))
fprintf('TX elements        = %d\n', size(txPos_16x16_1,1))
fprintf('Unique VA elements = %d\n', size(vaPos_16x16_1,1))

fprintf('\n16x16_2:\n')
fprintf('RX elements        = %d\n', size(rxPos_16x16_2,1))
fprintf('TX elements        = %d\n', size(txPos_16x16_2,1))
fprintf('Unique VA elements = %d\n', size(vaPos_16x16_2,1))


%% ============================================================
%  Local functions
% ============================================================

function vaPos = getVirtualArray(txPos, rxPos)

    M = size(txPos,1);
    N = size(rxPos,1);

    vaX = reshape(txPos(:,1), [M 1]) + reshape(rxPos(:,1), [1 N]);
    vaY = reshape(txPos(:,2), [M 1]) + reshape(rxPos(:,2), [1 N]);

    vaPosAll = [vaX(:), vaY(:)];

    % remove repeated VA elements
    vaPos = unique(round(vaPosAll,10), 'rows');

end


function plotPhysicalArray(txPos, rxPos, plotTitle)

    scatter(rxPos(:,1), rxPos(:,2), 55, 'filled')
    hold on
    scatter(txPos(:,1), txPos(:,2), 55, 'filled')

    grid on
    box on
    axis equal
    axis tight

    xlabel('$x/\lambda$', 'Interpreter','latex')
    ylabel('$y/\lambda$', 'Interpreter','latex')
    title(plotTitle)

    legend('RX','TX', 'Location','best')

end


function plotVirtualArray(vaPos, plotTitle)

    scatter(vaPos(:,1), vaPos(:,2), 22, 'filled')

    grid on
    box on
    axis equal
    axis tight

    xlabel('$x/\lambda$', 'Interpreter','latex')
    ylabel('$y/\lambda$', 'Interpreter','latex')
    title(plotTitle)

end