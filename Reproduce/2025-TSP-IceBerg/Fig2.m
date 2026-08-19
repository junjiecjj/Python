%% Fig2.m
% Faithful MATLAB translation of Fig2.py
clear; close all; clc; rng(42, 'twister');

set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultAxesFontSize',18);
set(groot,'defaultTextFontSize',18);
set(groot,'defaultLineLineWidth',2);
set(groot,'defaultLineMarkerSize',6);
set(groot,'defaultFigureColor','white');
set(groot,'defaultLegendFontSize',18);

%% Table I
M = 16;
N = 128;
L = 8;
alpha = 0.35;

M_array = [4, 16, 64, 256, 1024];
for M = M_array
    MOD_TYPE = 'qam';
    Constellation = qammod((0:M-1).',M,'gray','UnitAveragePower',true);
    kurtosis = mean(abs(Constellation).^4);
    fprintf('%d-%s, kurtosis = %.12f\n',M,upper(MOD_TYPE),kurtosis);
end

%% Fig. 2
Tsym = 1;
pi_value = pi;
N = 128;
L = 10;
alpha = 0.3;
span = 6;

%% 方法一：忠实对应原 Python，默认使用
[p1,t1,filtDelay1] = srrcFunction(alpha,L,span,Tsym);
p1 = [p1; zeros(L*N-numel(p1),1)];

%% 方法二：MATLAB 自带 rcosdesign
% p2 = rcosdesign(alpha,span,L,'sqrt').';
% t2 = (-span*Tsym/2:Tsym/L:span*Tsym/2).';
% filtDelay = (numel(p2)-1)/2;
% p2 = [p2;zeros(L*N-numel(p2),1)];

%% 方法三： Alternative corresponding to commpy.filters.rrcosfilter:
[p,t,filtDelay] = commpyRrcosfilter(L*N,alpha,Tsym,L/Tsym);
p = p/sqrt(sum(p.^2));

norm2p = norm(p);
FLN = FFTmatrix(L*N);
FN = FFTmatrix(N);

%% OFDM, Eq. (27) and Eq. (34)
M = 100;
kappa = 1.32;
U = FN';
V = eye(N);
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_Iceberg = zeros(L*N,1);
TheoAveACF_OFDM_M1 = zeros(L*N,1);
TheoAveACF_OFDM_M100 = zeros(L*N,1);
for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    r1 = L*N*abs(fk_tilde'*gk)^2;
    r2 = norm(gk)^2;
    r3 = (kappa-2)*L*N*norm(tilde_V*(gk.*conj(fk_tilde)))^2;

    TheoAveACF_OFDM_M1(k+1) = r1+(r2+r3)/1;
    TheoAveACF_OFDM_M100(k+1) = r1+(r2+r3)/100;
    TheoAveACF_Iceberg(k+1) = r1;
end

TheoAveACF_OFDM_M1 = TheoAveACF_OFDM_M1/max(TheoAveACF_OFDM_M1)+1e-10;
TheoAveACF_OFDM_M100 = TheoAveACF_OFDM_M100/max(TheoAveACF_OFDM_M100)+1e-10;
TheoAveACF_Iceberg = TheoAveACF_Iceberg/max(TheoAveACF_Iceberg)+1e-10;

x = (-N*L/2:N*L/2-1).';
figure('Position',[100,100,1200,1000]);
plot(x,10*log10(TheoAveACF_Iceberg),'k--','DisplayName','Squared ACF of the Pulse ("Iceberg")'); hold on;
plot(x,10*log10(TheoAveACF_OFDM_M1),'b-','DisplayName','Average Squared ACF, Theoretical');
plot(x,10*log10(TheoAveACF_OFDM_M100),'r-','DisplayName','100 Coherent Integration, Theoretical');
xlabel('Delay Index'); ylabel('Ambiguity Level (dB)');
legend('Location','best','EdgeColor','black'); 
box on; 
hold off;
drawnow;
pause(0.1);
%% OFDM, Eq. (36)
TheoAveACF_Iceberg = zeros(L*N,1);
TheoAveACF_OFDM_M1 = zeros(L*N,1);
TheoAveACF_OFDM_M100 = zeros(L*N,1);
for k = 0:L*N-1
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    fk = exp(-1j*2*pi_value*k*(0:N-1).'/(L*N));

    r1 = abs(gk.'*conj(fk))^2;
    TheoAveACF_Iceberg(k+1) = r1;

    M = 1;
    r2 = (kappa-1)/M*(N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
    TheoAveACF_OFDM_M1(k+1) = r1+r2;

    M = 100;
    r2 = (kappa-1)/M*(N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
    TheoAveACF_OFDM_M100(k+1) = r1+r2;
end

TheoAveACF_Iceberg = TheoAveACF_Iceberg/max(TheoAveACF_Iceberg)+1e-10;
TheoAveACF_Iceberg = fftshift(TheoAveACF_Iceberg);
TheoAveACF_OFDM_M1 = TheoAveACF_OFDM_M1/max(TheoAveACF_OFDM_M1)+1e-10;
TheoAveACF_OFDM_M1 = fftshift(TheoAveACF_OFDM_M1);
TheoAveACF_OFDM_M100 = TheoAveACF_OFDM_M100/max(TheoAveACF_OFDM_M100)+1e-10;
TheoAveACF_OFDM_M100 = fftshift(TheoAveACF_OFDM_M100);

x = (-N*L/2:N*L/2-1).';
figure('Position',[100,100,1200,1000]);
plot(x,10*log10(TheoAveACF_Iceberg),'k--','DisplayName','Squared ACF of the Pulse ("Iceberg")'); hold on;
plot(x,10*log10(TheoAveACF_OFDM_M1),'b-','DisplayName','Average Squared ACF, Theoretical');
plot(x,10*log10(TheoAveACF_OFDM_M100),'r-','DisplayName','100 Coherent Integration, Theoretical');
xlabel('Delay Index'); ylabel('Ambiguity Level (dB)'); xlim([-300,300]);
legend('Location','best','EdgeColor','black');
box on; 
hold off;
drawnow;
pause(0.1);

%% Average Squared ACF, Numerical / 100 Coherent Integration, Numerical, Eq. (26)
kappa = 1.32;
U = FN';
V = eye(N);
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

MOD_TYPE = 'qam';
Order = 16;
% qammod is called once. The loops index this stored constellation exactly
% as d = randint(...); s = Constellation[d] in the Python program.
Constellation = qammod((0:Order-1).',Order,'gray','UnitAveragePower',true);
AvgEnergy = mean(abs(Constellation).^2);

Iter = 1000;

%% M = 1
M = 1;
SimAveACF_OFDM_M1 = zeros(Iter,L*N);
for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    for it = 1:Iter
        d = randi([0,Order-1],N,1);
        s = Constellation(d+1);
        VHs = abs(V'*s).^2;
        SimAveACF_OFDM_M1(it,k+1) = abs(sum(gk.*VHs.*conj(fk_tilde)))^2;
    end
end

Sim_M1_avg = mean(SimAveACF_OFDM_M1,1);
Sim_M1_avg = Sim_M1_avg/max(Sim_M1_avg)+1e-10;
Sim_M1_avg = fftshift(Sim_M1_avg);
Sim_M1_max = max(SimAveACF_OFDM_M1,[],1);
Sim_M1_max = Sim_M1_max/max(Sim_M1_max)+1e-10;
Sim_M1_max = fftshift(Sim_M1_max);
Sim_M1_min = min(SimAveACF_OFDM_M1,[],1);
Sim_M1_min = Sim_M1_min/max(Sim_M1_min)+1e-10;
Sim_M1_min = fftshift(Sim_M1_min);

%% M = 100
M = 100;
SimAveACF_OFDM_M100 = complex(zeros(M,Iter,L*N));
for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    for m = 1:M
        for it = 1:Iter
            d = randi([0,Order-1],N,1);
            s = Constellation(d+1);
            VHs = abs(V'*s).^2;
            % Do not square here: average over M first, as in Eq. (33).
            SimAveACF_OFDM_M100(m,it,k+1) = sum(gk.*VHs.*conj(fk_tilde));
        end
    end
end

%% Eq. (33)
RkBar = mean(SimAveACF_OFDM_M100,1);
RkBar2 = abs(RkBar).^2;

Sim_M100_avg = squeeze(mean(RkBar2,2)).';
Sim_M100_avg = Sim_M100_avg/max(Sim_M100_avg)+1e-10;
Sim_M100_avg = fftshift(Sim_M100_avg);
Sim_M100_max = squeeze(max(RkBar2,[],2)).';
Sim_M100_max = Sim_M100_max/max(Sim_M100_max)+1e-10;
Sim_M100_max = fftshift(Sim_M100_max);
Sim_M100_min = squeeze(min(RkBar2,[],2)).';
Sim_M100_min = Sim_M100_min/max(Sim_M100_min)+1e-10;
Sim_M100_min = fftshift(Sim_M100_min);

colors = jet(5);
x = -N*L/2:N*L/2-1;
figure('Position',[100,100,1500,1000]);
plot(x,10*log10(TheoAveACF_Iceberg),'k-','LineWidth',1,'DisplayName','Squared ACF of the Pulse ("Iceberg")'); hold on;
plot(x,10*log10(TheoAveACF_OFDM_M1),'-','Color',[19,168,249]/255,'DisplayName','Average Squared ACF, Theoretical');
plot(x,10*log10(Sim_M1_avg),'-o','Color',colors(2,:),'LineWidth',1,'MarkerIndices',1:20:L*N,'MarkerSize',12,'MarkerFaceColor','none','DisplayName','Average Squared ACF, Simular');
fill([x,fliplr(x)],[10*log10(Sim_M1_min),fliplr(10*log10(Sim_M1_max))],[19,168,249]/255,'FaceAlpha',0.4,'EdgeColor','none','HandleVisibility','off');
plot(x,10*log10(TheoAveACF_OFDM_M100),'-','Color',[240,118,10]/255,'DisplayName','100 Coherent Integration, Theoretical');
plot(x,10*log10(Sim_M100_avg),'-o','Color',[249,114,19]/255,'LineWidth',1,'MarkerIndices',1:20:L*N,'MarkerSize',12,'MarkerFaceColor','none','DisplayName','100 Coherent Integration, Simular');
fill([x,fliplr(x)],[10*log10(Sim_M100_min),fliplr(10*log10(Sim_M100_max))],[240,118,10]/255,'FaceAlpha',0.4,'EdgeColor','none','HandleVisibility','off');
xlabel('Delay Index'); ylabel('Ambiguity Level (dB)'); xlim([-300,300]);
legend('Location','best','EdgeColor','black'); 

box on; 
hold off;
drawnow;

exportgraphics(gcf,'Fig2_1.png');
exportgraphics(gcf,'Fig2_1.pdf','ContentType','vector');

%% Local functions translated from Fig2.py

function mat = FFTmatrix(L)
    mat = complex(zeros(L,L));
    for i = 0:L-1
        for j = 0:L-1
            mat(i+1,j+1) = exp(-1j*2*pi*i*j/L)/sqrt(L);
        end
    end
end

function [p,t,filtDelay] = srrcFunction(beta,L,span,Tsym)
    t = (-span*Tsym/2:Tsym/L:span*Tsym/2).';
    A = sin(pi*t*(1-beta)/Tsym)+4*beta*t/Tsym.*cos(pi*t*(1+beta)/Tsym);
    B = pi*t/Tsym.*(1-(4*beta*t/Tsym).^2);
    p = 1/sqrt(Tsym)*A./B;
    p(isnan(p)) = 1;
    p(isinf(p)) = beta/sqrt(2*Tsym)*((1+2/pi)*sin(pi/(4*beta))+(1-2/pi)*cos(pi/(4*beta)));
    filtDelay = (numel(p)-1)/2;
    p = p/sqrt(sum(p.^2));
end

function [p,t,filtDelay] = commpyRrcosfilter(N,alpha,Ts,Fs)
    T_delta = 1/Fs;
    t = ((0:N-1).'-N/2)*T_delta;
    p = zeros(N,1);
    for x = 1:N
        t_x = t(x);
        if t_x == 0
            p(x) = 1-alpha+4*alpha/pi;
        elseif alpha ~= 0 && abs(t_x) == Ts/(4*alpha)
            p(x) = alpha/sqrt(2)*((1+2/pi)*sin(pi/(4*alpha))+(1-2/pi)*cos(pi/(4*alpha)));
        else
            p(x) = (sin(pi*t_x*(1-alpha)/Ts)+4*alpha*(t_x/Ts)*cos(pi*t_x*(1+alpha)/Ts))/(pi*t_x*(1-(4*alpha*t_x/Ts)^2)/Ts);
        end
    end
    filtDelay = (N-1)/2;
end