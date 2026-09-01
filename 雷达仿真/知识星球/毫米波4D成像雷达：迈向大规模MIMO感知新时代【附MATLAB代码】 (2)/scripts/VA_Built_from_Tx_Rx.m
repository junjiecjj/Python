clear
close all
clc

addpath(genpath('../functions/'))

num_elements = 8;
radius = 1;

theta_step = 2*pi / num_elements;

TX = zeros(num_elements, 2);

for i = 1:num_elements
    theta = (i-1) * theta_step;

    x = radius * cos(theta) + radius;
    y = radius * sin(theta) + radius;

    TX(i,:) = [x, y];
end

theta = pi/num_elements;

R = [cos(theta) -sin(theta); sin(theta) cos(theta)];

center = [radius, radius];

RX = zeros(size(TX));

for i = 1:num_elements
    point_translated = TX(i,:) - center;

    point_rotated = R * point_translated';

    RX(i,:) = point_rotated' + center;
end

RX = RX + [3*radius 3*radius];

%%

VA = computeVA(TX, RX);

ScreenSize                  = get(0,'ScreenSize');

fhdl(1)                     = figure('Name','TX RX Circ','NumberTitle','off',...
    'Position', [0 0 floor(ScreenSize(3)/4) floor(ScreenSize(4)/3)]);
movegui('northwest')
scatter(TX(:,1),TX(:,2),60,'o','LineWidth',1.5), hold on
scatter(RX(:,1),RX(:,2),60,'v','LineWidth',1.5)
legend('Tx Chain', 'Rx Chain', 'Interpreter','Latex', 'FontSize', 11, 'location', 'none')
xlabel('Horizental (Wavelength)','FontSize',11,'Interpreter','Latex')
ylabel('Vertical (Wavelength)','FontSize',11,'Interpreter','Latex')
TxRxMin = [min(min(TX(:,1)),min(RX(:,1))) min(min(TX(:,2)),min(RX(:,2)))];
TxRxMax = [max(max(TX(:,1)),max(RX(:,1))) max(max(TX(:,2)),max(RX(:,2)))];
xlim([floor(TxRxMin(1)*2)/2 ceil(TxRxMax(1)*2)/2])
ylim([floor(TxRxMin(2)*2)/2 ceil(TxRxMax(2)*2)/2])
xticks(floor(TxRxMin(1)*2)/2:0.5:ceil(TxRxMax(1)*2)/2);
yticks(floor(TxRxMin(2)*2)/2:0.5:ceil(TxRxMax(2)*2)/2);
box on
grid on

fhdl(2)                     = figure('Name','VA Circ','NumberTitle','off',...
    'Position', [0 0 floor(ScreenSize(3)/4) floor(ScreenSize(4)/3)]);
movegui('northeast')
VA(:,1) = VA(:,1) - min(VA(:,1));
VA(:,2) = VA(:,2) - min(VA(:,2));
scatter(VA(:,1),VA(:,2),60,'d','LineWidth',1.5,'MarkerEdgeColor','#7E2F8E')
xlabel('Horizental (Wavelength)','FontSize',11,'Interpreter','Latex')
ylabel('Vertical (Wavelength)','FontSize',11,'Interpreter','Latex')
VAMin = [min(VA(:,1)) min(VA(:,2))];
VAMax = [max(VA(:,1)) max(VA(:,2))];
xlim([floor(VAMin(1)*2)/2 ceil(VAMax(1)*2)/2])
ylim([floor(VAMin(2)*2)/2 ceil(VAMax(2)*2)/2]);
xticks(floor(VAMin(1)*2)/2:0.5:ceil(VAMax(1)*2)/2);
yticks(floor(VAMin(2)*2)/2:0.5:ceil(VAMax(2)*2)/2);
box on
grid on

%%

TX = [zeros(8,1) (0.5:0.5:4)'];
RX = [(0.5:0.5:4)' zeros(8,1)];

VA = computeVA(TX, RX);

fhdl(3)                     = figure('Name','TX RX Rect','NumberTitle','off',...
    'Position', [0 0 floor(ScreenSize(3)/4) floor(ScreenSize(4)/3)]);
movegui('southwest')
scatter(TX(:,1),TX(:,2),60,'o','LineWidth',1.5), hold on
scatter(RX(:,1),RX(:,2),60,'v','LineWidth',1.5)
legend('Tx Chain', 'Rx Chain', 'Interpreter','Latex', 'FontSize', 11, 'location', 'none')
xlabel('Horizental (Wavelength)','FontSize',11,'Interpreter','Latex')
ylabel('Vertical (Wavelength)','FontSize',11,'Interpreter','Latex')
TxRxMin = [min(min(TX(:,1)),min(RX(:,1))) min(min(TX(:,2)),min(RX(:,2)))];
TxRxMax = [max(max(TX(:,1)),max(RX(:,1))) max(max(TX(:,2)),max(RX(:,2)))];
xlim([floor(TxRxMin(1)*2)/2 ceil(TxRxMax(1)*2)/2])
ylim([floor(TxRxMin(2)*2)/2 ceil(TxRxMax(2)*2)/2])
xticks(floor(TxRxMin(1)*2)/2:0.5:ceil(TxRxMax(1)*2)/2);
yticks(floor(TxRxMin(2)*2)/2:0.5:ceil(TxRxMax(2)*2)/2);
box on
grid on

fhdl(4)                     = figure('Name','VA Rect','NumberTitle','off',...
    'Position', [0 0 floor(ScreenSize(3)/4) floor(ScreenSize(4)/3)]);
movegui('southeast')
VA(:,1) = VA(:,1) - min(VA(:,1));
VA(:,2) = VA(:,2) - min(VA(:,2));
scatter(VA(:,1),VA(:,2),60,'d','LineWidth',1.5,'MarkerEdgeColor','#7E2F8E')
xlabel('Horizental (Wavelength)','FontSize',11,'Interpreter','Latex')
ylabel('Vertical (Wavelength)','FontSize',11,'Interpreter','Latex')
VAMin = [min(VA(:,1)) min(VA(:,2))];
VAMax = [max(VA(:,1)) max(VA(:,2))];
xlim([floor(VAMin(1)*2)/2 ceil(VAMax(1)*2)/2])
ylim([floor(VAMin(2)*2)/2 ceil(VAMax(2)*2)/2]);
xticks(floor(VAMin(1)*2)/2:0.5:ceil(VAMax(1)*2)/2);
yticks(floor(VAMin(2)*2)/2:0.5:ceil(VAMax(2)*2)/2);
box on
grid on