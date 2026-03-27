function myresize(figure_name, h)
% myresize: resize the current figure to 7.5-by-12 inches
%   figure_name: the name of the saved eps figure

% eliminate margins
% set(gca, 'Position', get(gca, 'OuterPosition') - ...
%     get(gca, 'TightInset') * [-1 0 1 0; 0 -1 0 1; 0 0 1 0; 0 0 0 1]);

if nargin < 2
    h = gcf; 
end

% resize the figure
height = 7.5; % default is 6.25
width = 12; % default is 7.5
%height = 12.5; width = 15;
set(h, 'PaperUnits', 'inches');
set(h, 'PaperSize', [width height]);
set(h, 'PaperPositionMode', 'manual');
set(h, 'PaperPosition', [0 0 width height]);

%set(gcf, 'renderer', 'painters');

%print(h, '-depsc2', '-r300', [figure_name '.eps']);
print(h, '-deps', [figure_name '.eps']);
%print -depsc2 [figure_name '.eps']