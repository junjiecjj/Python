plot(FQPSK_QData(60:750),'k','linewidth',2);
axis([-60 690 -1.5 1.5]);
xlabel('t / T_s');ylabel('s_Q(t)');grid;
title('Quadrature  ( Delayed by T_s/2 )');
ax=gca;
ax.XTick=[-90:60:1140];
ax.XTickLabel={-2:12};

ax.YTick=[ -1 -.7071 0 .7071 1 ];
ax.YTickLabel{2}='-A';
ax.YTickLabel{4}='A';
ax.FontSize= 20;
ax.GridLineStyle= '-.';
ax.GridColor=[0 0 0];
line([0 0],[-1.5 0],'color','k','LineStyle','--','linewidth',1.5)

