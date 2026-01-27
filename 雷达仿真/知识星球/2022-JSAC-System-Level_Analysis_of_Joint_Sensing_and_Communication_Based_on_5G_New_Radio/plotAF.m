%计算并画出阵列天线方向图，计算-10dB主瓣宽度亦即单次波束扫描宽度(平均)
%%
function [Pattern, Delta_theta]=plotAF(phi,ifPlot,ifBeamWidth)%phi为天线最大辐射方向(角度制)
    global NT figureNumber;
    %NT=10;phi=50;ifPlot=0;ifBeamWidth=1;
    theta=-180:0.1:179.9;
    theta_radians=deg2rad(theta);
    phi_radians=deg2rad(phi);
    Pattern=abs((sin(NT*0.5*pi*(sin(theta_radians)-sin(phi_radians))))./(sin(0.5*pi*(sin(theta_radians)-sin(phi_radians)))));
    for theta=-180:0.1:179.9%处理阵列因子中的最大值点
        if ((theta-phi)==0 || theta+phi == 180 || theta+phi == -180)
            Pattern((phi+180)*10+1)=NT;
        end
    end
    
    if(ifPlot==1)
        %画方向图
        figure(figureNumber);
        figureNumber=figureNumber+1;
        polarplot(theta_radians, Pattern);
        text=["ULA Pattern with NT=NR="+num2str(NT)];
        title(text);
        str=['./fig/ArrayFactor'+num2str(figureNumber)+'.png'];
        saveas(gcf,str);
        close(gcf);
    end
    
    Delta_theta=0;
    if(ifBeamWidth==1)
        %计算（边射）-10dB波束宽度，仅当phi=0时
        maximum=NT;
        for i=((phi+180)*10+1):3600
            if(20*log10(maximum/Pattern(i))>=10)
                Delta_theta=2*((i-1)*0.1-phi-180);
                break;
            end
        end
    end
    
    Pattern = Pattern/NT; %归一化
    %Delta_theta=Delta_theta*1.5;
    %在扫描过程中，波束宽度会逐渐增大
    %增大的比例近似为1/cos(方向角)，当方向角为60°时，增大至2倍
end