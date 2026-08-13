function Figure_Launcher
% FIGURE_LAUNCHER  GUI to reproduce the figures of the 4D imaging mMIMO
% radars
%
%   Run it, check the figures you want to reproduce, and press
%   "Run Simulation". Multiple figures can be selected at once.
%
%% Locate package folders

rootDir = fileparts(mfilename('fullpath'));
simDir  = fullfile(rootDir, 'scripts');
funcDir = fullfile(rootDir, 'functions');
dataDir = fullfile(rootDir, 'data');

if ~isfolder(simDir)
    error('Figure_Launcher:badLocation', ...
        ['"scripts" folder not found. Place Figure_Launcher.m ', ...
         'in the root of the code package.']);
end

%% Figure list

entries = { ...
    'Table I  -  Effect of bandwidth and aperture ("4D" image sweep)', 'MIMO_FMCW_4D_Sweep.m',                      {}; ...
    'Fig. 6   -  Physical Tx/Rx arrays and their virtual arrays',      'VA_Built_from_Tx_Rx.m',                     {}; ...
    'Fig. 7a / 8a  -  Tx/Rx positions of commercial-style radars',     'VA_Position_of_Commercial_Radars.m',        {}; ...
    'Fig. 7b-d  -  Pedestrian point cloud (12x16 configuration)',      'Different_Array_Configuration_Point_Cloud.m', {'useArray = ''12x16'';', 'useArray = ''12x16'';'}; ...
    'Fig. 8b-d  -  Pedestrian point cloud (48x48 configuration)',      'Different_Array_Configuration_Point_Cloud.m', {'useArray = ''12x16'';', 'useArray = ''48x48'';'}; ...
    'Fig. 19  -  Ambiguity functions of FMCW and PMCW',                'AF_for_FMCW_and_PMCW.m',                    {}; ...
    'Fig. 20  -  MIMO vs. phased array beampatterns',                  'MIMO_vs_Phased_Array.m',                    {}; ...
    'Fig. 21  -  2D beampatterns of URA configurations',               'Two_D_BP.m',                                {}; ...
    'Fig. 24  -  Reference image (walking person, pyramid, sphere)',   'Imaging_Properties.m',                      {}; ...
    'Fig. 25  -  SAR image output',                                    'NF_SAR_Imaging.m',                          {}; ...
    'Fig. 26  -  4D massive MIMO radar image output',                  'Four_D_Radar_Imaging.m',                    {}; ...
    };

nEntries = size(entries, 1);

%% Build GUI

rowH     = 27;
margin   = 14;
btnH     = 32;
figW     = 520;
figH     = margin + btnH + margin + nEntries*rowH + 46;

fh = figure('Name', 'mmWave 4D Imaging Survey - Figure Launcher', ...
    'NumberTitle',      'off', ...
    'MenuBar',          'none', ...
    'ToolBar',          'none', ...
    'Resize',           'off', ...
    'HandleVisibility', 'off', ...
    'Color',            get(0, 'defaultUicontrolBackgroundColor'), ...
    'Position',         [0 0 figW figH]);
movegui(fh, 'center');

uicontrol(fh, 'Style', 'text', ...
    'String',              'Select the figure(s) or table to reproduce:', ...
    'FontSize',            10, ...
    'FontWeight',          'bold', ...
    'HorizontalAlignment', 'left', ...
    'Position',            [margin, figH - 34, figW - 2*margin, 22]);

cbh = gobjects(nEntries, 1);
for k = 1:nEntries
    yPos = figH - 40 - k*rowH;
    cbh(k) = uicontrol(fh, 'Style', 'checkbox', ...
        'String',   ['  ', entries{k,1}], ...
        'FontSize', 9.5, ...
        'Value',    0, ...
        'Position', [margin + 6, yPos, figW - 2*margin - 6, rowH - 4]);
end

uicontrol(fh, 'Style', 'pushbutton', ...
    'String',   'Select All', ...
    'FontSize', 9, ...
    'Position', [margin, margin, 90, btnH], ...
    'Callback', @(~,~) set(cbh, 'Value', 1));

uicontrol(fh, 'Style', 'pushbutton', ...
    'String',   'Clear All', ...
    'FontSize', 9, ...
    'Position', [margin + 100, margin, 90, btnH], ...
    'Callback', @(~,~) set(cbh, 'Value', 0));

uicontrol(fh, 'Style', 'pushbutton', ...
    'String',     'Run Simulation', ...
    'FontSize',   10, ...
    'FontWeight', 'bold', ...
    'Position',   [figW - margin - 150, margin, 150, btnH], ...
    'Callback',   @(~,~) onRun());

%% ------------------------------------------------------------------
    function onRun()
        selected = find(cellfun(@(v) isequal(v, 1), get(cbh, 'Value')));
        if isempty(selected)
            return
        end

        delete(fh);
        drawnow

        oldDir     = pwd;
        restoreDir = onCleanup(@() cd(oldDir));     
        cd(simDir);
        addpath(funcDir, dataDir);

        protectedFigs = gobjects(0, 1);

        for idx = selected(:).'

            scriptFile = fullfile(simDir, entries{idx, 2});
            patchPair  = entries{idx, 3};
            tempFile   = '';

            if ~isempty(patchPair) && ~strcmp(patchPair{1}, patchPair{2})
                txt = fileread(scriptFile);
                txt = strrep(txt, patchPair{1}, patchPair{2});
                tempFile = fullfile(simDir, ['tmp_launcher_', entries{idx, 2}]);
                fid = fopen(tempFile, 'w');
                fwrite(fid, txt);
                fclose(fid);
                scriptFile = tempFile;
            end

            try
                runIsolated(scriptFile);
            catch ME
                warning('Figure_Launcher:scriptFailed', ...
                    'Script "%s" failed: %s', entries{idx, 2}, ME.message);
            end

            if ~isempty(tempFile) && isfile(tempFile)
                delete(tempFile);
            end

            newFigs = findall(0, 'Type', 'figure');
            set(newFigs, 'HandleVisibility', 'off');
            protectedFigs = unique([protectedFigs; newFigs(:)]);
        end

        protectedFigs = protectedFigs(isgraphics(protectedFigs));
        set(protectedFigs, 'HandleVisibility', 'on');
    end

end

%% ------------------------------------------------------------------
function runIsolated(scriptPath)
run(scriptPath);
end
