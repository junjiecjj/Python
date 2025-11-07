if ischar(params) || isstring(params)
    configTable = readtable(params);
    result = cell(height(configTable), 1);
    for i = 1:height(configTable)
        fprintf("Running configuration %d/%d...\n", i, height(configTable));
        row = configTable(i, :);
        paramStruct = table2struct(row);
        result{i} = run_isac_simulation(paramStruct);
    end
    return;
end