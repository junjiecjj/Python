function [] = startmatlabpool(size)
 
p = gcp('nocreate');    % p获取现在有没有并行池在运行
if isempty(p)
    poolsize = 0;
else
    poolsize = p.NumWorkers     % poolsize 是工作室的数目
end

if poolsize == 0    % 如果poolsize==0则需要创建，使用parpool('local',size)来创建
    if nargin == 0
        parpool('local');
    else
        try
            parpool('local',size);
        catch ce
            parpool;
            fail_p = gcp('nocreate');
            fail_size = fail_p.NumWorkers;
            display(ce.message);
            display(strcat('输入的size不正确，采用的默认配置size=',num2str(fail_size)));
        end
    end
else
    display('parpool start');   % 当然如果你电脑是双核的你把size设置为3，那就出异常了，为了处理这个异常情况，此时应该忽略输入的size，直接使用本地默认的配置
    if poolsize ~= size
        closematlabpool();
        startmatlabpool(size);
    end
end

% 还用一种情况就是你当前已经有工作池在运行但是你还是要创建，那就只能把当前的关闭了然后再创建
