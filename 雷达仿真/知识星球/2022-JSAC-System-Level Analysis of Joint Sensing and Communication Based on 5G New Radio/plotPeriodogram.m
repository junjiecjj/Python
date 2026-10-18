function plotPeriodogram(range,velocity,P_TauDoppler,str)
    global figureNumber;
    Kp=length(range);
    Mp=length(velocity);
    velocity_1=kron(ones(Kp,1),velocity);
    range_1=kron(ones(1,Mp),range.');

    f=figure(figureNumber);
    f.OuterPosition=[100,50,800,720];

    subplot(2,2,1) %投影速度谱
    mesh(range_1,velocity_1,P_TauDoppler);
    xlabel('r / m');
    ylabel('v / m·s^-^1');
    zlabel('P');
    ylim([0 60]);
    xlim([10 90]);
    view(90,0)
    title(["P(v), \theta ∈ "+str]);

    subplot(2,2,2) %投影距离谱
    mesh(range_1,velocity_1,P_TauDoppler);
    xlabel('r / m');
    ylabel('v / m·s^-^1');
    zlabel('P');
    ylim([0 60]);
    xlim([10 90]);
    view(0,0);
    title(["P(r), \theta ∈ "+str]);

    subplot(2,2,3) %距离-速度
    mesh(range_1,velocity_1,P_TauDoppler);
    xlabel('r / m');
    ylabel('v / m·s^-^1');
    zlabel('P');
    ylim([0 60]);
    xlim([10 90]);
    title(["P(r,v) top view, \theta ∈ "+str]);
    view(0,90);

    subplot(2,2,4) %3D视角
    mesh(range_1,velocity_1,P_TauDoppler);
    xlabel('r / m');
    ylabel('v / m·s^-^1');
    zlabel('P');
    ylim([0 60]);
    xlim([10 90]);
    title(["P(r,v), \theta ∈ "+str]);

    str=['./fig/Figure ',num2str(figureNumber),'_Periodogram.png'];
    saveas(gcf, str);
    close(gcf);
    figureNumber=figureNumber+1;
end

