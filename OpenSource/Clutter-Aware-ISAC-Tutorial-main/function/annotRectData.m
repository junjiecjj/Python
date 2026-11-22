function h = annotRectData(ax, x, y, wData, hData, varargin)
% Draw a rectangle annotation using data coordinates.
%
% Inputs:
%   ax       : Target axes handle
%   x, y     : Center of the rectangle in data coordinates
%   wData    : Rectangle width in data units
%   hData    : Rectangle height in data units
%   varargin : Additional properties passed to annotation
%
% Output:
%   h        : Handle to the rectangle annotation

fig = ancestor(ax, 'figure');
% Convert the center point from data coordinates to normalized figure coordinates
[xn, yn] = localData2Norm(ax, x, y);

% Convert the horizontal and vertical extents to normalized figure units
[xr, ~] = localData2Norm(ax, x + wData/2, y);
[xl, ~] = localData2Norm(ax, x - wData/2, y);
[~, yu] = localData2Norm(ax, x, y + hData/2);
[~, yd] = localData2Norm(ax, x, y - hData/2);
wN = abs(xr - xl);
hN = abs(yu - yd);

% Annotation position in normalized figure coordinates
pos = [xn - wN/2, yn - hN/2, wN, hN];
h = annotation(fig, 'rectangle', pos, varargin{:});
end