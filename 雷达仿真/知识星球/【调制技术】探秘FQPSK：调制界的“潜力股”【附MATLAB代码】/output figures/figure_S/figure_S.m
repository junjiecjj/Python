b= linspace(-ts/2,ts/2,60);
plot(b,s7,'k','linewidth',2);title('S_7(t) = - S_1_5(t)');grid;
A=0.7071;
axis([-1.5e-6 1.5e-6 -1.3 1.3])
line([-ts/2 -ts/2],[-1.3 -1],'color','k','LineStyle','--','linewidth',1.5)
line([ts/2 ts/2],[-1.3 1],'color','k','LineStyle','--','linewidth',1.5)
ax=gca;
ax.XTick=[-1.0417e-6 -.5e-6 0 .5e-6 1.0417e-6];
ax.YTick=[-1 -.7071 0 .7071 1];
ax.XTickLabel{1}='-T_s/2';
ax.XTickLabel{end}='T_s/2';
ax.YTickLabel{2}='-A';
ax.YTickLabel{end-1}='A';
ax.FontSize= 20;
ax.GridLineStyle= '-.';
xlabel('time(\mus)');
ylabel('A =1/\surd2    ')