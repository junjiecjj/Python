function [time,PRN,para] = rinex303Reader(f)
fid = fopen(f);
while ~feof(fid)
    t = fgetl(fid);
    if contains(t,'END OF HEADER')
        break;
    end
end
ii = 0;
PRN = NaN(24*63,1);
time = NaT(24*63,1);
para = NaN(24*63,29);
while ~feof(fid)
    ii = ii + 1;
    PRN(ii) = fscanf(fid,'C%2f',1);
    t = fscanf(fid,'%4f%2f%2f%2f%2f%2f',6);
    time(ii) = datetime(t','Format','yyyy-MM-dd HH:mm:ss');
    para(ii,:) = fscanf(fid,'%E\n',29)';
end
fclose(fid);
time = time(1:ii);
PRN = PRN(1:ii);
para = para(1:ii,:);