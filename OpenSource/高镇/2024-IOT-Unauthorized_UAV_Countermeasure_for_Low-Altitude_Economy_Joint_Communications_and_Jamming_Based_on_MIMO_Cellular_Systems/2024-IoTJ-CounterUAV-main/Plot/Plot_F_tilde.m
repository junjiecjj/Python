close all
%%
load("..\Data\F_tilde_WithUE.mat")
set(0,'defaultfigurecolor','w') 
figure;
F_tilde = abs(F_tilde)/max(abs(F_tilde), [], 'all');
surf(F_tilde)
xlim([1 size(F_tilde,2)]);
ylim([1 size(F_tilde,1)]);
colormap parula
shading interp
colorbar
set(gca,'FontName','Times New Roman','FontSize',16);
%%
load("..\Data\F_tilde_WithoutUE.mat")
set(0,'defaultfigurecolor','w') 
figure;
F_tilde = abs(F_tilde)/max(abs(F_tilde), [], 'all');
surf(F_tilde)
xlim([1 size(F_tilde,2)]);
ylim([1 size(F_tilde,1)]);
colormap parula
shading interp
colorbar
set(gca,'FontName','Times New Roman','FontSize',16);
%%
load("..\Data\F_tilde_SDP.mat")
set(0,'defaultfigurecolor','w') 
figure;
F_tilde = abs(F_tilde)/max(abs(F_tilde), [], 'all');
surf(F_tilde)
xlim([1 size(F_tilde,2)]);
ylim([1 size(F_tilde,1)]);
colormap parula
shading interp
colorbar
set(gca,'FontName','Times New Roman','FontSize',16);

set(0,'defaultfigurecolor','w') 
figure;
F = abs(f*f')/max(abs(f*f'), [], 'all');
surf(F)
xlim([1 size(F,2)]);
ylim([1 size(F,1)]);
colormap parula
shading interp
colorbar
set(gca,'FontName','Times New Roman','FontSize',16);