%% Fig. 6(d) -- corrected reproduction
% Uncovering the Iceberg in the Sea: Fundamentals of Pulse Shaping and
% Modulation Design for Random ISAC Signals
%
% The delay region [5,15] in Fig. 6 is expressed in symbol-delay units.
% Since the discrete ACF is sampled at Ts = T/L, the optimization uses
% the sample-delay set K_s1 = 5*L:15*L.
%
% CVX is required: http://cvxr.com/cvx/

clear;
clc;
close all;

rng(42);

Tsym = 1;
N = 128;
L = 10;
alpha = 0.35;

%% Pulse and Fourier matrices
[p,t,filtDelay] = commpyRrcosfilter(L*N,alpha,Tsym,L/Tsym); 
p = p/sqrt(sum(p.^2));

norm2p = norm(p); 
FLN = FFTmatrix(L*N);
FN = FFTmatrix(N); 

%% Squared spectrum of the designed pulse
% Paper delay region: 5 <= tau/T <= 15.
% Discrete ACF sample index: k = tau/Ts = L*tau/T.
K_s1 = 5*L:15*L;

gN = solve_iceberg_shaping_psl(N,L,alpha,K_s1);

fprintf('Optimal spectrum coefficients:\n');
disp(gN);

g_N = 1-gN;
g_design = [gN; zeros((L-2)*N,1); g_N];
g_Design = fftshift(g_design);

%% Squared spectrum of the RRC pulse
M = 100; 
kappa = 1.32; 
g = N*(FLN*p).*(conj(FLN)*conj(p));
g_rrc = fftshift(real(g));

%% Numerical verification of (45)--(48) in the adopted FFT ordering
N_alpha = fix(alpha*N);
N_non_rolloff = N-N_alpha;
N_zeros = floor(N_non_rolloff/2);
N_ones = floor(N_non_rolloff/2);

err_fixed_one = max(abs(gN(1:N_zeros)-1));
err_fixed_zero = max(abs(gN(N-N_ones+1:N)));
err_monotonic = max([0; diff(gN)]);
err_energy = abs(sum(gN)-N/2);
rolloff = gN(N_zeros+1:N-N_ones);

fprintf('Fixed-one constraint error:     %.3e\n',err_fixed_one);
fprintf('Fixed-zero constraint error:    %.3e\n',err_fixed_zero);
fprintf('Monotonic constraint error:     %.3e\n',err_monotonic);
fprintf('Energy constraint error:        %.3e\n',err_energy);
fprintf('Roll-off dynamic range:         %.6f\n',max(rolloff)-min(rolloff));

if max(rolloff)-min(rolloff) < 1e-3
    error(['The optimized roll-off region collapsed to an approximately ', 'constant value. Check the delay-index scaling and CVX result.']);
end

%% Plot Fig. 6(d)
x = (-N*L/2:N*L/2-1).';

figure('Color','w','Position',[100,100,760,560]);
hold on;

plot(x,abs(g_Design),'--','Color',[0.8500,0.3250,0.0980], 'LineWidth',1.8,'DisplayName','Spectrum of the Designed Pulse');
plot(x,abs(g_rrc),'-','Color',[0,0.4470,0.7410],  'LineWidth',1.8,'DisplayName','Spectrum of the RRC');

legend('Location','northwest','Box','on');
xlabel('Frequency Index');
ylabel('Power Spectrum');
xlim([-N,N]);
ylim([0,1.2]);
xticks(-100:50:100);
yticks(0:0.2:1.2);
grid on;
box on;

set(gca,'FontName','Times New Roman','FontSize',14,'LineWidth',1);

exportgraphics(gcf,'Fig6_d_correct.png','Resolution',300);
exportgraphics(gcf,'Fig6_d_correct.pdf','ContentType','vector');


function g_opt = solve_iceberg_shaping_psl(N,L,alpha,K_s1)
    % PSL iceberg-shaping problem in (44) and (49).
    
    N_alpha = fix(alpha*N);
    N_non_rolloff = N-N_alpha;
    N_zeros = floor(N_non_rolloff/2);
    N_ones = floor(N_non_rolloff/2);
    
    cvx_begin quiet
        variable g(N) nonnegative
        expressions psl_terms(length(K_s1))
    
        for indexK = 1:length(K_s1)
            k = K_s1(indexK);
            f_k = exp(-1j*2*pi*k*(0:N-1).'/(L*N));
            gk = g+(1-g)*exp(-1j*2*pi*k/L);
            psl_terms(indexK) = square_abs(f_k'*gk);
        end
    
        minimize(max(psl_terms))
    
        subject to
            % Frequency ordering used by FFTmatrix and fftshift:
            % 1 -> monotonically decreasing roll-off -> 0.
            g(1:N_zeros) == 1;
            g(N-N_ones+1:N) == 0;
    
            for n = 1:N-1
                g(n+1)-g(n) <= 0;
            end
    
            sum(g) == N/2;
    cvx_end
    
    if strcmp(cvx_status,'Solved') || strcmp(cvx_status,'Inaccurate/Solved')
        fprintf('Optimization successful!\n');
        fprintf('Optimal PSL value: %.12e\n',cvx_optval);
        g_opt = g;
    else
        error('CVX failed to solve the PSL problem. Status: %s',cvx_status);
    end
end


function mat = FFTmatrix(L)
    mat = complex(zeros(L,L));
    ll = 0:L-1;
    for i = 0:L-1
        mat(i+1,:) = exp(-1j*2*pi*i*ll/L)/sqrt(L);
    end
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
                ((1+2/pi)*sin(pi/(4*alpha))+ ...
                (1-2/pi)*cos(pi/(4*alpha)));
        elseif alpha ~= 0 && t_x == -Ts/(4*alpha)
            p(x) = alpha/sqrt(2)* ...
                ((1+2/pi)*sin(pi/(4*alpha))+ ...
                (1-2/pi)*cos(pi/(4*alpha)));
        else
            p(x) = ...
                (sin(pi*t_x*(1-alpha)/Ts)+ ...
                4*alpha*(t_x/Ts)*cos(pi*t_x*(1+alpha)/Ts))/ ...
                (pi*t_x*(1-(4*alpha*t_x/Ts)^2)/Ts);
        end
    end
    
    filtDelay = (N-1)/2;
end