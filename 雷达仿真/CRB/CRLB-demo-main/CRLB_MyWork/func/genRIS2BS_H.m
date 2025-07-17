function H = genRIS2BS_H(xR1, yR1,...
                        xRN, yRN,...
                        xB1, yB1,...
                        xBN, yBN,...
                        NRIS, NBS, fc)
%% 生成RIS到BS的信道                    
% Inputs:
%       RIS和基站的头和尾的坐标值
%       RIS和BS的天线数
%       fc                              =       载频
% Outputs：
%       RIS 2 BS 的信道
%%
H = zeros(NBS, NRIS);
% 线阵各个阵元的坐标的生成
xB = linspace(xB1, xBN, NBS);
yB = linspace(yB1, yBN, NBS);
% xR = linspace(xR1, xRN, NRIS);
% yR = linspace(yR1, yRN, NRIS);
for i_NBS = 1:NBS
    H(i_NBS, :) = gen1toArray(xB(i_NBS), yB(i_NBS),...
                   xR1, yR1,...
                   xRN, yRN,...
                   NRIS, fc).';
end

end