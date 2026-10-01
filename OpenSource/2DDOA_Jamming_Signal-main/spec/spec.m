clear;
clc;
close all;
Nt=500;
%%nested
M1c=3;
M2c=M1c+1;
dx_0=sort([0:M1c:(M2c-1)*M1c,M2c:M2c:(2*M1c-1)*M2c])';
K=9;
Mf=max(dx_0)+1;
Ih=eye(Mf);
a_s=Ih(dx_0+1,:);
%%coprime
% M1c=3;
% M2c=M1c+1;
%sort([0:M1c:(M2c-1)*M1c,M2c:M2c:(2*M1c-1)*M2c])';
% M1n=4;
% M2n=M1n;
% dx_co=[0:M1n-1,M1n:M1n+1:M2n*(M1n+1)-1]';
%%差分阵参数生成
[dx,w]=w_arr_cal(dx_0);
[dc,c,c_p,c_n]=cent_ula(dx);
M=length(dx_0);
D=ceil(length(dx)/2);
U=ceil(length(dc)/2);
dxp=dx(D:end)';
dcp=dc(U:end)';
% fun_A=@(p1,p2)[A_th(p1,dcp);A_th(p1,dcp)*exp(1j*p2)];
fun_A=@(p1,p2)[A_th(p1,(0:U-1)');A_th(p1,(0:U-1)')*exp(1j*p2)];
fun_B=@(p1,p2)[A_th(p1,dx_0);A_th(p1,dx_0)*exp(1j*p2)];
%%
omega0=-0.7*pi+(0:K-1)*2*pi/14+0*randn(1,K);%-0.7812*pi+(0:K-1)*0.108*pi+0*randn(1,K);
omega0=(mod(omega0/pi+1,2)-1)*pi;
ccc=randperm(K);
omega0=omega0(ccc);
gamma0=-0.7*pi+(0:K-1)*2*pi/12+0*randn(1,K);
gamma0=(mod(gamma0/pi+1,2)-1)*pi;
A=A_th(omega0,dx_0);
B=diag(exp(1j*gamma0));
%%参数设置
Snrdb=10;
Snr=10.^(Snrdb./20);
SnrB=Snrdb(1);
ss=length(Snr);
Ps=diag(ones(1,K));
%%
Sa_set=false(1,Mf);
Sa_set(dx_0+1)=1;
D_sa=zeros(Mf,Mf);
D_sa(Sa_set,Sa_set)=1;
Sa_set1=logical(kron(ones(1,2),Sa_set));
D_sa1=zeros(Mf*2,Mf*2);
D_sa1(Sa_set1,Sa_set1)=1;
gg = 100;
for j=1:gg
    nng1{j}=1/sqrt(2)*(randn(M,Nt)+1j*randn(M,Nt));
    nng2{j}=1/sqrt(2)*(randn(M,Nt)+1j*randn(M,Nt));
end

for j = 1:gg
    S=Ps*1/sqrt(2)*(randn(K,Nt)+1j*randn(K,Nt));
    %         S=Ps*sqrt(Nt)*orth((1/sqrt(2)*(randn(K,Nt)+1j*randn(K,Nt)))')';
    y1=A*S+1/(Snr)*nng1{j};%阵列1
    y2=A*B*S+1/(Snr)*nng2{j};%阵列2
    Rg=[A;A*B]*Ps*Ps*[A;A*B]'+1/(Snr)^2*eye(2*M);
    
    R11=y1*y1'/Nt;
    R21=y2*y1'/Nt;
    R22=y2*y2'/Nt;
    Rt0=[R11,R21';R21,R22];
    Rt=size(Rt0,1)*Rt0/trace(Rt0);
    
    %差分阵输出
    r11=v_arr_cal(y1*y1'/Nt,dx_0);
    r22=v_arr_cal(y2*y2'/Nt,dx_0);
    r21=v_arr_cal(y2*y1'/Nt,dx_0);
    Ru11=toeplitz(r11(c_p),r11(c_n));
    Ru21=toeplitz(r21(c_p),r21(c_n));
    Ru22=toeplitz(r22(c_p),r22(c_n));
    Ru=[Ru11,Ru21';Ru21,Ru22];
    [M_omega(j,:),M_gamma(j,:)] = D2MUSIC(Ru,fun_A,K);
    [R_omega(j,:),R_gamma(j,:)]=F2D_MUSIC(Ru,U,K);
    [P_omega(j,:),P_gamma(j,:)]=P_PM(Ru11,Ru22,Ru21,1/Snr,K);
    
    % savefig('Spec_CAB.fig');saveas(gca,['Spec_CAB.eps'],'epsc');
    
    Rh=mat_sqrt(Rt);
    Ri=Rt^-1;
    R0=SPA_R(Sa_set1,Ri,2,Mf);
    R0 = [ R0(1:15,1:15) R0(1:15,22:36);R0(22:36,1:15) R0(22:36,22:36)];
    [L1_omega(j,:),L1_gamma(j,:)]=IP_PM(R0,U,K);
%     D_M_o(j)=sum(abs(sort(M_omega)-sort(omega0)))/K;
%     D_M_g(j)=sum(abs(sort(M_gamma)-sort(gamma0)))/K;
%     D_P_o(j)=sum(abs(sort(P_omega)-sort(omega0)))/K;
%     D_P_g(j)=sum(abs(sort(P_gamma)-sort(gamma0)))/K;
%     
%     D_R_o(j)=sum(abs(sort(R_omega)-sort(omega0)))/K;
%     D_R_g(j)=sum(abs(sort(R_gamma)-sort(gamma0)))/K;
%     
%     D_L1_o(j)=sum(abs(sort(L1_omega)-sort(omega0)))/K;
%     D_L1_g(j)=sum(abs(sort(L1_gamma)-sort(gamma0)))/K;
end
figure;
subplot(221);hold on;box on;
h3 =plot(P_omega(1),P_gamma(1),'.','Color','#EDB120','Markersize',12);%legend('True','PM');
plot(P_omega(1),P_gamma(1),'.','Color','#FFFFFF','Markersize',12);
plot(omega0,gamma0,'ks')
plot(P_omega,P_gamma,'.','Color','#EDB120');%legend('True','PM');
xlabel('\omega','FontSize',14,'Fontname','Times New Roman','FontWeight','bold');
ylabel('\gamma','FontSize',14,'Fontname','Times New Roman','FontWeight','bold');

subplot(222);hold on;box on;
h1 =plot(M_omega(1),M_gamma(1),'.','Color','#77AC30','Markersize',12);%legend('True','MUSIC');
plot(M_omega(1),M_gamma(1),'.','Color','#FFFFFF','Markersize',12);
plot(omega0,gamma0,'ks');
plot(M_omega,M_gamma,'.','Color','#77AC30');%legend('True','MUSIC');
h0 = plot(omega0(1),gamma0(1),'ks');
xlabel('\omega','FontSize',14,'Fontname','Times New Roman','FontWeight','bold');
ylabel('\gamma','FontSize',14,'Fontname','Times New Roman','FontWeight','bold');

subplot(223);hold on;box on;
h2 =plot(R_omega(1),R_gamma(1),'.','Color','#7E2F8E','Markersize',12);%legend('True','RMUSIC');
plot(R_omega(1),R_gamma(1),'.','Color','#FFFFFF','Markersize',12);
plot(omega0,gamma0,'ks')
plot(R_omega,R_gamma,'.','Color','#7E2F8E');%legend('True','RMUSIC');
xlabel('\omega','FontSize',14,'Fontname','Times New Roman','FontWeight','bold');
ylabel('\gamma','FontSize',14,'Fontname','Times New Roman','FontWeight','bold');

subplot(224);hold on;box on;
h4 =plot(L1_omega(1),L1_gamma(1),'.','Color','#0072BD','Markersize',12);%legend('True','Proposed');
plot(L1_omega(1),L1_gamma(1),'.','Color','#FFFFFF','Markersize',12);%legend('True','Proposed');
plot(omega0,gamma0,'ks');
plot(L1_omega,L1_gamma,'.','Color','#0072BD');%legend('True','Proposed');
xlabel('\omega','FontSize',14,'Fontname','Times New Roman','FontWeight','bold');
ylabel('\gamma','FontSize',14,'Fontname','Times New Roman','FontWeight','bold');

legend([h0,h3,h1,h2,h4],'True','PM','MUSIC','RMUSIC','Proposed','FontSize',12,'Fontname','Times New Roman','NumColumns',5)
savefig('Sample.fig');saveas(gca,['Sample.eps'],'epsc');
% MUSIC = (D_M_o + D_M_g)/2;
% RootMUSIC = (D_R_o + D_R_g)/2;
% PM = (D_P_o + D_P_g)/2;
% Our = (D_L1_o + D_L1_g)/2;
% 
% MAX_MUSIC = max(MUSIC); MIN_MUSIC = min(MUSIC);RMSE_MUSIC = mean(MUSIC);
% MAX_RootMUSIC = max(RootMUSIC); MIN_RootMUSIC = min(RootMUSIC);RMSE_RootMUSIC = mean(RootMUSIC);
% MAX_PM = max(PM); MIN_PM = min(PM);RMSE_PM = mean(PM);
% MAX_Our = max(Our); MIN_Our = min(Our);RMSE_Our = mean(Our);
% 
% Mean = [RMSE_MUSIC RMSE_RootMUSIC RMSE_PM RMSE_Our];
% Min = [MIN_MUSIC MIN_RootMUSIC MIN_PM MIN_Our];
% Max = [MAX_MUSIC MAX_RootMUSIC MAX_PM MAX_Our];
% 
% 
% figure; hold on;box on;grid on;
% errorbar((1:4)+0.1,Mean,Mean - Min, Max-Min,'o' ,'linewidth',1.2);
% ylabel('Estimation Error','FontSize',14,'Fontname','Times New Roman','FontWeight','bold');
% labels = {'2D MUSIC','2D RMUSIC','2D PM','Proposed'};
% xticks(1:length(labels));
% xticklabels(labels);
% ax = gca; % 获取当前轴的句柄
% ax.FontSize = 14; % 设置字体大小
% ax.FontWeight = 'bold'; % 设置字体为粗体
% ax.FontName = 'Times New Roman'; % 设置字体类型为Arial
% set(ax, 'YScale', 'log');
% ylim([0.005 0.1]);xlim([1 4.2]);
% savefig('Sample_error.fig');saveas(gca,['Sample_error.eps'],'epsc');