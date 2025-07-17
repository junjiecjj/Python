figure; hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',15);
load('CDL_Near.mat')
plot(P_t_dBm, real(CRLB), 'ro-', 'LineWidth',1.5);

load('CDL_Far.mat')
plot(P_t_dBm, real(CRLB), 'r<-', 'LineWidth',1.5);

load('PDL_Near.mat')
plot(P_t_dBm, real(CRLB), 'go-', 'LineWidth',1.5);

load('PDL_Far.mat')
plot(P_t_dBm, real(CRLB), 'g<-', 'LineWidth',1.5);

load('CDL_Near.mat')
plot(P_t_dBm, real(CRLB), 'sm', 'LineWidth',1.5, 'MarkerSize', 12);

set(gca, 'YScale', 'log')
set(gca, 'YLim',[5e-8 1e-4]);
ylabel('CRLB$_{\vartheta}^{0.5}$','Interpreter','latex','fontweight','bold')
xlabel('Pt/dBm','Interpreter','latex','fontweight','bold')
legend('CDL Near', 'CDL Far', 'PDL Near', 'PDL Far', 'CDL Near Plus')