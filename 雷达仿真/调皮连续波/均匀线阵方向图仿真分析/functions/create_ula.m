function ula = create_ula(nr, d)
% CREATE_ULA 创建均匀线阵结构体
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 创建包含阵元数量和归一化间距的均匀线阵结构体
% 版本: 1.0
% 输入参数:
%   nr - 阵元数量
%   d  - 归一化单元间距(相对波长)，默认0.5
% 输出参数:
%   ula - 均匀线阵结构体

    if nargin < 2
        d = 0.5;
    end
    ula.nr = nr;
    ula.d = d;
end