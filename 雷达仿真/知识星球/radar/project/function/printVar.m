function printVar(varargin)
% 打印传入的若干个数据的名字和数据
for i = 1:length(varargin)
    % 打印数据名字
    if  ~isempty(inputname(i))
        fprintf('%s:\n', inputname(i));
    end
    disp(varargin{i});
    fprintf('\n');
end
end
