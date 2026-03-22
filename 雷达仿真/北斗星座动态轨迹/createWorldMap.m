
function ax = createWorldMap
figure;
ax = worldmap('World');
%setm(ax,'MapProjection','eqdcylin')
setm(ax, 'Origin', [0 0 0]);
ax.Position =[0 0 1 1];
set(gcf,'WindowState','maximized','MenuBar','none');
land = shaperead('landareas', 'UseGeoCoords', true);
geoshow(ax, land, 'FaceColor', [0.5 0.7 0.5]);
lakes = shaperead('worldlakes', 'UseGeoCoords', true);
geoshow(lakes, 'FaceColor', 'blue');
rivers = shaperead('worldrivers', 'UseGeoCoords', true);
geoshow(rivers, 'Color', 'blue');
end