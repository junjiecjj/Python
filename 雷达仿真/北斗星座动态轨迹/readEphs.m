function ephTable = readEphs
fileList = dir('2022/*.22b');
n = numel(fileList);
timeCell = cell(n,1);
PRNCell = cell(n,1);
paraCell = cell(n,1);
for ii  = 1:n
    [timeCell{ii},PRNCell{ii},paraCell{ii}] = rinex303Reader(fullfile(fileList(ii).folder,fileList(ii).name));
end
time = cat(1,timeCell{:});
PRN = cat(1,PRNCell{:});
para = cat(1,paraCell{:});
ephTable = timetable(time,PRN,para);
end