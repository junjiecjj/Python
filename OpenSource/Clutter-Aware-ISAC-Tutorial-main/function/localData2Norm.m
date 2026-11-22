function [xn, yn] = localData2Norm(ax, x, y)
% Convert a point from axes data coordinates to normalized figure coordinates.
%
% Inputs:
%   ax   : Target axes handle
%   x, y : Point in data coordinates
%
% Outputs:
%   xn   : x-coordinate in normalized figure units
%   yn   : y-coordinate in normalized figure units

fig = ancestor(ax, 'figure');
% Temporarily switch to normalized units
oldAxUnits = ax.Units;
ax.Units = 'normalized';
oldFigUnits = fig.Units;
fig.Units = 'normalized';
axpos = ax.Position;   % Axes position in normalized figure coordinates
xl = ax.XLim;
yl = ax.YLim;

% Convert x-coordinate
xn = axpos(1) + (x - xl(1)) / (xl(2) - xl(1)) * axpos(3);

% Convert y-coordinate, accounting for reversed y-axis if needed
if strcmpi(ax.YDir, 'reverse')
    yn = axpos(2) + (yl(2) - y) / (yl(2) - yl(1)) * axpos(4);
else
    yn = axpos(2) + (y - yl(1)) / (yl(2) - yl(1)) * axpos(4);
end

% Restore original units
ax.Units = oldAxUnits;
fig.Units = oldFigUnits;
end