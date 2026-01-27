function plotGridfull(prsGrid, dataGrid,DMRSGrid,PTRSGrid)
    %   plotGrid(PRSGRID, DATAGRID) plots the carrier grid for a single gNB.

    numgNBs = 1;  % Assuming only one gNB
    figure()
    mymap = [1 1 1; ...       % White color for background
             0 1 0; ...       % Green color for PDSCH data
             0 0 1;...      % Blue color for PRS data
             1 0 0];      % red color for DMRS
               

    gridWithPRS = abs(prsGrid{1});
    gridWithData = abs(dataGrid{1});
    gridWithDMRS =  abs(DMRSGrid{1});
    gridWithPTRS = abs(PTRSGrid{1});

    gridWithData = gridWithData - gridWithDMRS - gridWithPTRS;
    % Replace all zero values of gridWithPRS with a proper scaling (value of 2)
    % for PRS data visibility
    % gridWithPRS(gridWithPRS ~= 0) = 3;
    % gridWithPRS(gridWithPRS == 0) = 1;
    % 
    % gridWithData(gridWithData ~= 0) = 1;
    % gridWithDMRS(gridWithDMRS ~= 0) = 3;
    % gridWithPTRS(gridWithPTRS ~= 0) = 4;


    gridWithPRS(gridWithPRS ~= 0) = 2;
    gridWithPRS(gridWithPRS == 0) = 1;

    gridWithData(gridWithData ~= 0) = 1;
    gridWithDMRS(gridWithDMRS ~= 0) = 3;







    % Plot grid
    image(gridWithPRS + gridWithData + gridWithDMRS + gridWithPTRS); % In the resultant grid, 1 represents
                                       % the white background, 2 represents PRS,
                                       % and 3 represents PDSCH

    % Apply colormap
    colormap(mymap);
    axis xy;

    % Generate lines
    L = line(ones(3), ones(3), 'LineWidth', 8);
    % Index colormap and associate selected colors with lines
    set(L, {'color'}, mat2cell(mymap(2:end, :), ones(1, 3), 3));

    % Create legend
    legend('Data', 'PRS + Data','DMRS');

    % Add title
    title('Carrier Grid Containing PRS and PDSCH');

    % Add labels to x-axis and y-axis
    xlabel('OFDM Symbols');
    ylabel('Subcarriers');
end
