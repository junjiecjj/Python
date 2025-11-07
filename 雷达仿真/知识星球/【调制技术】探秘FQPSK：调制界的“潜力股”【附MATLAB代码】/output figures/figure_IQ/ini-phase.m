plot(FQPSK_IData(),'k','linewidth',2);
axis([-60 660 -1.5 1.5]);
xlabel('t / T_s');ylabel('s_I(t)');grid;
title('Ini-phase');
ax=gca;
ax.XTick=[-60:60:1140];
ax.XTickLabel={-1.5:1:11.5};

ax.YTick=[ -1 -.7071 0 .7071 1 ];
ax.YTickLabel{2}='-A';
ax.YTickLabel{4}='A';
ax.FontSize= 20;
ax.GridLineStyle= '-.';
ax.GridColor=[0 0 0];


