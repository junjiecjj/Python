function [clutter_data] = generate_lgclutter(radar_points, sigmav, muc, sigmac)
    fr = 1000;%脉冲重复频率
    lambda = 0.05;%波长
    sigmaf = 2*sigmav/lambda;%多普勒频移

    %生成高斯随机分布序列同相正交分量
    rand('state',sum(100*clock));%产生服从 0-1的均匀分布随机序列
    d1 = rand(1,radar_points);
    rand('state',7*sum(100*clock)+1);
    d2 = rand(1,radar_points);
    xi = 2*sqrt(-2*log(d1)).*cos(2*pi*d2);

    %滤波器设计：傅里叶级数展开
    coe_num = 12;
    for n = 0:coe_num
        coeff(n+1) = 2*sigmaf*sqrt(pi)*exp(-4*sigmaf^2*pi^2*n^2/fr^2)/fr;
    end
    for n = 1:2*coe_num+1
        if n <= coe_num+1
            b(n) = 1/2*coeff(coe_num+2-n);
        else
            b(n) = 1/2*coeff(n-coe_num);
        end
    end

    %生成高斯谱杂波
    xxi = conv(b,xi);
    xxi = xxi(coe_num*2+1:radar_points+coe_num*2);
    xsigmac = std(xxi);
    xmuc = mean(xxi);
    yyi = (xxi-xmuc)/xsigmac;%标准归一化0-1高斯

    %变换生成对数正态分布数据
    yyi = sigmac*yyi+log(muc);
    clutter_data = exp(yyi);%生成对数正态分布的杂波序列
end