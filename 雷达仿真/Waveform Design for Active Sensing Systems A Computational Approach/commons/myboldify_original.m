function myboldify_original(h)
% myboldify: make lines and text bold
%   myboldify boldifies the current figure
%   myboldify(h) applies to the figure with the handle h

if nargin < 1
    h = gcf; 
end

ha = get(h, 'Children'); % the handle of each axis

% Get the paper position of the current figure
units = get(gcf, 'PaperUnits');
set(gcf, 'PaperUnits', 'inches');
p_inch = get(gcf, 'PaperPosition'); % [left bottom width height] of the whole figure, in inch
set(gcf, 'PaperUnits', units);

for i = 1:length(ha)
    hc = get(ha(i), 'Children'); % the objects within an axis
    for j = 1:length(hc)
        chtype = get(hc(j), 'Type');
        if strcmp(chtype(1:4), 'text')
            set(hc(j), 'FontSize', 18); % 14 pt descriptive labels
            set(hc(j), 'FontWeight', 'Bold');
        elseif strcmp(chtype(1:4), 'line')
            set(hc(j), 'LineWidth', 2);
        end
    end
    if strcmp(get(ha(i),'Type'), 'axes') % axis format
        % Determine the scale in inches
        units = get(ha(i), 'Units');
        set(ha(i), 'Units', 'normalized');
        p_norm = get(ha(i), 'Position'); % normalized [left bottom width height] of this axis, 
                                         % with respect to the father figure
        set(ha(i), 'Units', units);
        scale = 1/(p_norm(3)*p_inch(3)); % 1/(true width)

        set(ha(i), 'FontSize', 18);      % tick mark and frame format
        set(ha(i), 'FontWeight', 'Bold');
        set(ha(i), 'LineWidth', 2);
        % set(ha(i), 'TickLength', [1/8 2.5*1/8]*scale); % [2D 3D], (1/8)"
        % setting "ticklength" causes trouble in colorbar

        set(get(ha(i),'XLabel'), 'FontSize', 18);
        set(get(ha(i),'XLabel'), 'FontWeight', 'Bold');
        set(get(ha(i),'XLabel'), 'VerticalAlignment', 'top');

        set(get(ha(i),'YLabel'), 'FontSize', 18);
        set(get(ha(i),'YLabel'), 'FontWeight', 'Bold');
        set(get(ha(i),'YLabel'), 'VerticalAlignment', 'baseline');

        set(get(ha(i),'ZLabel'), 'FontSize', 18);
        set(get(ha(i),'ZLabel'), 'FontWeight', 'Bold');
        set(get(ha(i),'ZLabel'), 'VerticalAlignment', 'baseline');

        set(get(ha(i),'Title'), 'FontSize', 18);
        set(get(ha(i),'Title'), 'FontWeight', 'Bold');
    end
end