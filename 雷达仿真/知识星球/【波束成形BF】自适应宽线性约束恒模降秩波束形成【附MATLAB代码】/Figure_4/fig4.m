clc
clear
%% SETTINGS
d = 3;
m = linspace(10,80,8);
%% OPERATIONS
struc_WLCCM_KS = 16*d*m.^2 + 20*m.^2 + 41*d*m + 4*m + 5*d^2 + 9*d - 2;
dir_WLCCM_KS = 32*d*m.^2 + 40*m.^2 + 16*d*m + 56*m + 5*d^2 + 4*d - 2;
WL_AVF = 64*d*m.^2 + 40*m.^2 + 72*d*m + 8*m;
LCCM_KS = 8*d*m.^2 + 10*m.^2 + 24*d*m + 8*m + 18*d^2 + 29*d + 1;
L_AVF = 16*d*m.^2 + 10*m.^2 + 36*d*m;
%% PLOT
semilogy(m,struc_WLCCM_KS,'LineWidth', 1, 'DisplayName',"WLCCM-KS (Structured)")
hold on
semilogy(m,dir_WLCCM_KS,'LineWidth', 1, 'DisplayName',"WLCCM-KS (Direct)")
hold on
semilogy(m,WL_AVF,'LineWidth', 1, 'DisplayName',"WL-AVF")
hold on
semilogy(m,LCCM_KS,'LineWidth', 1, 'DisplayName',"LCCM-KS")
hold on
semilogy(m,L_AVF,'LineWidth', 1, 'DisplayName',"L-AVF")
hold on
xlabel("Length of Received Vector M")
ylabel("Number of Real Operations")
title("Number of real operations per snapshot versus length of received vector M")
legend show
grid on
