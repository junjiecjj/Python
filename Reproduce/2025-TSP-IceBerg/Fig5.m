%% Fig5.m
% Faithful MATLAB translation of Fig5.py
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

% Active pulse-generation method in Fig5.py:
[p,t,filtDelay] = commpyRrcosfilter(L*N,alpha,Tsym,L/Tsym);
p = p/sqrt(sum(p.^2));

norm2p = norm(p);
FLN = FFTmatrix(L*N);
FN = FFTmatrix(N);

%% OFDM, 16QAM, Eq. (36), simplified expression
kappa = 1.32;
U = FN';
V = eye(N);
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_OFDM_16QAM_M1 = zeros(L*N,1);
for k = 0:L*N-1
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    fk = exp(-1j*2*pi_value*k*(0:N-1).'/(L*N));

    r1 = abs(gk.'*conj(fk))^2;

    M = 1;
    r2 = (kappa-1)/M* ...
        (N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
    TheoAveACF_OFDM_16QAM_M1(k+1) = r1+r2;
end

TheoAveACF_OFDM_16QAM_M1 = ...
    TheoAveACF_OFDM_16QAM_M1/max(TheoAveACF_OFDM_16QAM_M1)+1e-10;
TheoAveACF_OFDM_16QAM_M1 = fftshift(TheoAveACF_OFDM_16QAM_M1);

%% OFDM, 1024QAM, Eq. (36), simplified expression
kappa = 1.3988;
U = FN';
V = eye(N);
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_OFDM_1024QAM_M1 = zeros(L*N,1);
for k = 0:L*N-1
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    fk = exp(-1j*2*pi_value*k*(0:N-1).'/(L*N));

    r1 = abs(gk.'*conj(fk))^2;

    M = 1;
    r2 = (kappa-1)/M* ...
        (N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
    TheoAveACF_OFDM_1024QAM_M1(k+1) = r1+r2;
end

TheoAveACF_OFDM_1024QAM_M1 = ...
    TheoAveACF_OFDM_1024QAM_M1/max(TheoAveACF_OFDM_1024QAM_M1)+1e-10;
TheoAveACF_OFDM_1024QAM_M1 = fftshift(TheoAveACF_OFDM_1024QAM_M1);

%% OFDM, PSK, Eq. (36), simplified expression
kappa = 1;
U = FN';
V = eye(N);
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_Iceberg = zeros(L*N,1);
TheoAveACF_OFDM_PSK_M1 = zeros(L*N,1);
for k = 0:L*N-1
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    fk = exp(-1j*2*pi_value*k*(0:N-1).'/(L*N));

    r1 = abs(gk.'*conj(fk))^2;
    TheoAveACF_Iceberg(k+1) = r1;

    M = 1;
    r2 = (kappa-1)/M* ...
        (N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
    TheoAveACF_OFDM_PSK_M1(k+1) = r1+r2;
end

TheoAveACF_Iceberg = TheoAveACF_Iceberg/max(TheoAveACF_Iceberg)+1e-10;
TheoAveACF_Iceberg = fftshift(TheoAveACF_Iceberg);
TheoAveACF_OFDM_PSK_M1 = ...
    TheoAveACF_OFDM_PSK_M1/max(TheoAveACF_OFDM_PSK_M1)+1e-10;
TheoAveACF_OFDM_PSK_M1 = fftshift(TheoAveACF_OFDM_PSK_M1);

%% OFDM, Gaussian, Eq. (36), simplified expression
kappa = 2;
U = FN';
V = eye(N);
tilde_V = V.*conj(V);
g = N*(FLN*p).*(conj(FLN)*conj(p));

TheoAveACF_OFDM_Gaussian_M1 = zeros(L*N,1);
for k = 0:L*N-1
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    fk = exp(-1j*2*pi_value*k*(0:N-1).'/(L*N));

    r1 = abs(gk.'*conj(fk))^2;

    M = 1;
    r2 = (kappa-1)/M* ...
        (N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
    TheoAveACF_OFDM_Gaussian_M1(k+1) = r1+r2;
end

TheoAveACF_OFDM_Gaussian_M1 = ...
    TheoAveACF_OFDM_Gaussian_M1/max(TheoAveACF_OFDM_Gaussian_M1)+1e-10;
TheoAveACF_OFDM_Gaussian_M1 = fftshift(TheoAveACF_OFDM_Gaussian_M1);

%% Plot together
colors = jet(5); %#ok<NASGU>
x = -N*L/2:N*L/2-1;
figure('Position',[100,100,1200,800]);
% The following Iceberg curve is commented in Fig5.py and is retained here:
% plot(x,10*log10(TheoAveACF_Iceberg),'k--', ...
%     'DisplayName','Squared ACF of the Pulse ("Iceberg")'); hold on;

plot(x,10*log10(TheoAveACF_OFDM_16QAM_M1),'b--', ...
    'DisplayName','OFDM, 16QAM, Theoretical');
hold on;
plot(x,10*log10(TheoAveACF_OFDM_1024QAM_M1),'r-', ...
    'DisplayName','OFDM, 1024QAM, Theoretical');
plot(x,10*log10(TheoAveACF_OFDM_PSK_M1),'k--', ...
    'DisplayName','OFDM, PSK, Theoretical');
plot(x,10*log10(TheoAveACF_OFDM_Gaussian_M1),'g--', ...
    'DisplayName','OFDM, Gaussian, Theoretical');

legend('Location','best','EdgeColor','black');
xlabel('Delay Index');
ylabel('Ambiguity Level (dB)');
xlim([-200,200]);
box on;
hold off;
drawnow;

exportgraphics(gcf,'Fig5.png');
exportgraphics(gcf,'Fig5.pdf','ContentType','vector');

%% Local functions translated from Fig5.py
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
    A = sin(pi*t*(1-beta)/Tsym)+ ...
        4*beta*t/Tsym.*cos(pi*t*(1+beta)/Tsym);
    B = pi*t/Tsym.*(1-(4*beta*t/Tsym).^2);
    p = 1/sqrt(Tsym)*A./B;
    p(isnan(p)) = 1;
    p(isinf(p)) = beta/sqrt(2*Tsym)* ...
        ((1+2/pi)*sin(pi/(4*beta))+(1-2/pi)*cos(pi/(4*beta)));
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
    elseif alpha ~= 0 && t_x == Ts/(4*alpha)
        p(x) = alpha/sqrt(2)* ...
            ((1+2/pi)*sin(pi/(4*alpha))+(1-2/pi)*cos(pi/(4*alpha)));
    elseif alpha ~= 0 && t_x == -Ts/(4*alpha)
        p(x) = alpha/sqrt(2)* ...
            ((1+2/pi)*sin(pi/(4*alpha))+(1-2/pi)*cos(pi/(4*alpha)));
    else
        p(x) = ...
            (sin(pi*t_x*(1-alpha)/Ts)+ ...
            4*alpha*(t_x/Ts)*cos(pi*t_x*(1+alpha)/Ts))/ ...
            (pi*t_x*(1-(4*alpha*t_x/Ts)^2)/Ts);
    end
end
filtDelay = (N-1)/2;
end
