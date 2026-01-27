function colors = getColors(numgNBs)
%    COLORS = getColors(NUMGNBS) returns the RGB triplets COLORS for the
%    plotting purpose, given the number of gNBs NUMGNBS.

    % Form RGB triplets
    if numgNBs <= 10
        colors = [0.8500, 0.3250, 0.0980; 0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560; ...
                  0.4660, 0.6740, 0.1880; 0.3010, 0.7450, 0.9330; 0.6350, 0.0780, 0.1840; ...
                  0.9290, 0.9, 0.3; 0.9290, 0.5, 0.9; 0.6660, 0.3740, 0.2880;0, 0.4470, 0.7410];
    else
        % Generate 30 more extra RGB triplets than required. It is to skip
        % the gray and white shades as these shades are reserved for the
        % data and no transmissions in carrier grid plot in this example.
        % With this, you can get unique colors for up to 1000 gNBs.
        colors = colorcube(numgNBs+30);
        colors = colors(1:numgNBs,:);
    end
end