%% Fig3.m
% Faithful MATLAB translation of Fig3.py
clear; close all; clc; rng(42,'twister');

set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultAxesFontSize',18);
set(groot,'defaultTextFontSize',18);
set(groot,'defaultLineLineWidth',2);
set(groot,'defaultLineMarkerSize',6);
set(groot,'defaultFigureColor','white');
set(groot,'defaultLegendFontSize',18);

%% Parameter settings
Tsym = 1;
pi_value = pi;
N = 128;
L = 10;
alpha = 0.3;
% span = 6;

% Alternative corresponding to the commented Python srrcFunction code:
% [p,t,filtDelay] = srrcFunction(alpha,L,span,Tsym);
% p = [p;zeros(L*N-numel(p),1)];

% Active pulse-generation method in Fig3.py:
[p,t,filtDelay] = commpyRrcosfilter(L*N,alpha,Tsym,L/Tsym);
p = p/sqrt(sum(p.^2));

norm2p = norm(p);
FLN = FFTmatrix(L*N);
FN = FFTmatrix(N);

%% OFDM, Eq. (36), simplified expression
kappa = 1.32;
U = FN';
V = eye(N);
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_Iceberg = zeros(L*N,1);
TheoAveACF_OFDM_M1 = zeros(L*N,1);

for k = 0:L*N-1
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    fk = exp(-1j*2*pi_value*k*(0:N-1).'/(L*N));

    r1 = abs(gk.'*conj(fk))^2;
    TheoAveACF_Iceberg(k+1) = r1;

    M = 1;
    r2 = (kappa-1)/M* ...
        (N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
    TheoAveACF_OFDM_M1(k+1) = r1+r2;
end

TheoAveACF_Iceberg = TheoAveACF_Iceberg/max(TheoAveACF_Iceberg)+1e-10;
TheoAveACF_Iceberg = fftshift(TheoAveACF_Iceberg);
TheoAveACF_OFDM_M1 = TheoAveACF_OFDM_M1/max(TheoAveACF_OFDM_M1)+1e-10;
TheoAveACF_OFDM_M1 = fftshift(TheoAveACF_OFDM_M1);

%% SC, Eq. (27) and Eq. (34), general expression
kappa = 1.32;
U = eye(N);
V = U'*FN';
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_SC_M1 = zeros(L*N,1);
for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    r1 = L*N*abs(fk_tilde'*gk)^2;
    r2 = norm(gk)^2;
    r3 = (kappa-2)*L*N*norm(tilde_V*(gk.*conj(fk_tilde)))^2;

    TheoAveACF_SC_M1(k+1) = r1+(r2+r3)/1;
end

TheoAveACF_SC_M1 = TheoAveACF_SC_M1/max(TheoAveACF_SC_M1)+1e-10;
TheoAveACF_SC_M1 = fftshift(TheoAveACF_SC_M1);

%% SC, Eq. (37), commented alternative retained from Fig3.py
% kappa = 1.32;
% g = N*(FLN*p).*(conj(FLN)*conj(p));
% TheoAveACF_SC_M1 = zeros(L*N,1);
% for k = 0:L*N-1
%     gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
%     fk = exp(-1j*2*pi_value*k*(0:N-1).'/(L*N));
%     M = 1;
%     r1 = (1+(kappa-2)/(M*N))*abs(gk.'*conj(fk))^2;
%     r2 = 1/M*(N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
%     TheoAveACF_SC_M1(k+1) = r1+r2;
% end
% TheoAveACF_SC_M1 = TheoAveACF_SC_M1/max(TheoAveACF_SC_M1)+1e-10;
% TheoAveACF_SC_M1 = fftshift(TheoAveACF_SC_M1);

%% SC, Eq. (26), 1000 Monte Carlo, general expression
kappa = 1.32;
U = eye(N);
V = U'*FN';
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

MOD_TYPE = 'qam';
Order = 16;
Constellation = qammod((0:Order-1).',Order,'gray','UnitAveragePower',true);
AvgEnergy = mean(abs(Constellation).^2);

Iter = 1000;
M = 1;
SimAveACF_SC_M1 = zeros(Iter,L*N);

for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    for it = 1:Iter
        d = randi([0,Order-1],N,1);
        s = Constellation(d+1);
        VHs = abs(V'*s).^2;
        SimAveACF_SC_M1(it,k+1) = ...
            abs(sum(gk.*VHs.*conj(fk_tilde)))^2;
    end
end

Sim_SC_M1_avg = mean(SimAveACF_SC_M1,1);
Sim_SC_M1_avg = Sim_SC_M1_avg/max(Sim_SC_M1_avg)+1e-10;
Sim_SC_M1_avg = fftshift(Sim_SC_M1_avg);

%% CDMA, Eq. (27) and Eq. (34), general expression
kappa = 1.32;
U = hadamard_matrix_sylvester(N)/sqrt(N);
V = U'*FN';
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_CDMA_M1 = zeros(L*N,1);
for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    r1 = L*N*abs(fk_tilde'*gk)^2;
    r2 = norm(gk)^2;
    r3 = (kappa-2)*L*N*norm(tilde_V*(gk.*conj(fk_tilde)))^2;

    TheoAveACF_CDMA_M1(k+1) = r1+(r2+r3)/1;
end
TheoAveACF_CDMA_M1 = abs(TheoAveACF_CDMA_M1);
TheoAveACF_CDMA_M1 = TheoAveACF_CDMA_M1/max(TheoAveACF_CDMA_M1)+1e-10;
TheoAveACF_CDMA_M1 = fftshift(TheoAveACF_CDMA_M1);

%% OTFS, Eq. (27) and Eq. (34), general expression
kappa = 1.32;
FFTN = 32;
Neye = fix(N/FFTN);
FFTM = FFTmatrix(FFTN);
eyeM = eye(Neye);
U = kron(FFTM,eyeM);
V = U'*FN';
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_OTFS_M1 = zeros(L*N,1);
for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    r1 = L*N*abs(fk_tilde'*gk)^2;
    r2 = norm(gk)^2;
    r3 = (kappa-2)*L*N*norm(tilde_V*(gk.*conj(fk_tilde)))^2;

    TheoAveACF_OTFS_M1(k+1) = r1+(r2+r3)/1;
end
TheoAveACF_OTFS_M1 = abs(TheoAveACF_OTFS_M1);
TheoAveACF_OTFS_M1 = TheoAveACF_OTFS_M1/max(TheoAveACF_OTFS_M1)+1e-10;
TheoAveACF_OTFS_M1 = fftshift(TheoAveACF_OTFS_M1);

%% AFDM, Eq. (27) and Eq. (34), general expression
c1 = 1/128;
c2 = 4/(3*pi_value);
U = IDAFT(c1,c2,N);
V = U'*FN';
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_AFDM_M1 = zeros(L*N,1);
for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    r1 = L*N*abs(fk_tilde'*gk)^2;
    r2 = norm(gk)^2;
    r3 = (kappa-2)*L*N*norm(tilde_V*(gk.*conj(fk_tilde)))^2;

    TheoAveACF_AFDM_M1(k+1) = r1+(r2+r3)/1;
end
TheoAveACF_AFDM_M1 = abs(TheoAveACF_AFDM_M1);
TheoAveACF_AFDM_M1 = TheoAveACF_AFDM_M1/max(TheoAveACF_AFDM_M1)+1e-10;
TheoAveACF_AFDM_M1 = fftshift(TheoAveACF_AFDM_M1);

%% Plot together
colors = jet(5);
x = -N*L/2:N*L/2-1;
figure('Position',[100,100,1200,800]);
plot(x,10*log10(TheoAveACF_Iceberg),'k--','DisplayName','Squared ACF of the Pulse ("Iceberg")'); hold on;
plot(x,10*log10(TheoAveACF_OFDM_M1),'b-','DisplayName','OFDM, M = 1, Theoretical');
plot(x,10*log10(TheoAveACF_SC_M1),'r-','DisplayName','SC, M = 1, Theoretical');
plot(x,10*log10(Sim_SC_M1_avg),'r--o','MarkerIndices',1:20:L*N,'MarkerSize',12,'MarkerFaceColor','none','LineWidth',1,'DisplayName','SC, M = 1, Simulat');
plot(x,10*log10(TheoAveACF_CDMA_M1),'-','Color',colors(3,:),'DisplayName','CDMA, M = 1, Theoretical');
plot(x,10*log10(TheoAveACF_OTFS_M1),'--*','Color',colors(1,:),'MarkerIndices',1:20:L*N,'MarkerSize',12,'MarkerFaceColor','none','LineWidth',1,'DisplayName','OTFS, M = 1, Theoretical');
plot(x,10*log10(TheoAveACF_AFDM_M1),'--s','Color',[0.5,0,0.5],'MarkerIndices',1:20:L*N,'MarkerSize',12,'MarkerFaceColor','none','LineWidth',1,'DisplayName','AFDM, M = 1, Theoretical');
legend('Location','best','EdgeColor','black');
xlabel('Delay Index'); ylabel('Ambiguity Level (dB)'); xlim([-200,200]);
box on; hold off; drawnow;
exportgraphics(gcf,'Fig3.png');
exportgraphics(gcf,'Fig3.pdf','ContentType','vector');

%% Local functions translated from Fig3.py
function mat = FFTmatrix(L)
mat = complex(zeros(L,L));
ll = 0:L-1;
for i = 0:L-1
    mat(i+1,:) = exp(-1j*2*pi*i*ll/L)/sqrt(L);
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

function H = hadamard_matrix_sylvester(n)
if n == 1
    H = 1;
else
    H_prev = hadamard_matrix_sylvester(n/2);
    H = kron(H_prev,[1,1;1,-1]);
end
end

function U = IDAFT(c1,c2,N)
F = fft(eye(N));
F = F/norm(F,2);
n = 0:N-1;
L1 = diag(exp(-1j*2*pi*c1*(n.^2)));
L2 = diag(exp(-1j*2*pi*c2*(n.^2)));
A = L2*F*L1;
U = A';
end

function [p,t,filtDelay] = commpyRrcosfilter(N,alpha,Ts,Fs)
T_delta = 1/Fs;
t = ((0:N-1).'-N/2)*T_delta;
p = zeros(N,1);
for x = 1:N
    t_x = t(x);
    if t_x == 0
        p(x) = 1-alpha+4*alpha/pi;
    elseif alpha ~= 0 && t_x == Ts/(4*alpha)
        p(x) = alpha/sqrt(2)*((1+2/pi)*sin(pi/(4*alpha))+(1-2/pi)*cos(pi/(4*alpha)));
    elseif alpha ~= 0 && t_x == -Ts/(4*alpha)
        p(x) = alpha/sqrt(2)*((1+2/pi)*sin(pi/(4*alpha))+(1-2/pi)*cos(pi/(4*alpha)));
    else
        p(x) = (sin(pi*t_x*(1-alpha)/Ts)+4*alpha*(t_x/Ts)*cos(pi*t_x*(1+alpha)/Ts))/(pi*t_x*(1-(4*alpha*t_x/Ts)^2)/Ts);
    end
end
filtDelay = (N-1)/2;
end
