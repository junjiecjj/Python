

clear;
clc;
close all;

rng(42,'twister');

% num of BS antennas
N = 64;
% num of receiver antennas
M = 16;
% num of users
K = 1;
% num of data streams per user
d = 6;
% num of data streams
Ns = K * d;
Nrf = Ns;
% stopping (convergence) condition
epsilon = 1e-4;
L = 15;

sigma2 = 40;
% num of iterations for each dB step
num_iters = 1000;

SNRdBs = -10:2:6;
RateLst = zeros(1,length(SNRdBs));
parfor i = 1:length(SNRdBs)
    i
    snrdB = SNRdBs(i);
    P = 10^(snrdB / 10) * sigma2;
    for it = 1:num_iters
        H = channel(K, N, M, L);
        H = squeeze(H);
        tmp = Algo2(H, Ns, P, sigma2, epsilon);
        RateLst(i) = RateLst(i) + tmp;
    end
    RateLst(i) = RateLst(i) / num_iters;
end

figure(1);
% 使用简写格式绘制
plot(SNRdBs, RateLst, 'b--o', ...
    'LineWidth', 2, ...
    'MarkerSize', 8, ...
    'DisplayName', 'Single-User Large-Scale MIMO');
xlabel('SNR(dB)', 'FontSize', 16);
ylabel('Spectral Efficiency(bits/s/Hz)', 'FontSize', 16);
grid on;
legend('FontSize', 16);
hold on;


function Rate = Algo2(H, Ns, P, sigma2, epsilon)
    Rate = 0;
    Nrf = Ns;
    [M, N] = size(H);
    gamma = sqrt(P / (N * Nrf));
    % Find Vrf using alg1
    F_1 = H' * H;
    Vrf = alg1(F_1, Ns, gamma^2, sigma2, epsilon);
    
    [N, Nrf1] = size(Vrf);
    % Find Ue and GAMMAe matrices (11)
    Heff = H * Vrf;
    Q = Vrf' * Vrf;
    % Right singular vectors, 
    [U, S, V] = svd(Heff * sqrtm(pinv(Q)));
    s_values = diag(S);
    Ue = V';
    % Diagonal matrix of allocated powers to each stream
    GAMMAe = eye(Nrf) * (P/Nrf)^0.5;

    % Computing digital precoder matrix (11)
    Vd =  sqrtm(pinv(Q)) * Ue * GAMMAe;
    % Vd = (np.linalg.inv(Q)**(1/2) @ Ue @ GAMMAe).astype(np.complex128)

    % Hybrid precoder matrix (8)
    Vt = Vrf * Vd;

    % Compute analog combiner matrix of receiver (15)
    F_2 =  H * (Vt * Vt') * H';
    Wrf = alg1(F_2, Ns, 1/M, sigma2, epsilon);

    % Compute the digital combiner matrix of receiver (17)
    J = Wrf' * H * (Vt * Vt') * H' * Wrf  + sigma2 * (Wrf' * Wrf);
    Wd = pinv(J) * Wrf' * H * Vt;

    % Hybrid combiner matrix (8)
    Wt = Wrf * Wd;

    % Compute the spectral efficiency metric (4)
    Rate = log2(det(real( eye(M) + 1/sigma2 * H * Vrf * (Vd * Vd') * Vrf' * H')));

end

function Vrf = alg1(F, Ns, gamma2, sigma2, epsilon)
    Nrf = Ns;
    N = size(F, 1);
    Vrf = ones(N, Nrf);
    last_iter_obj = 0.0;
    iter_obj = 0.0;
    diff = 1.0;
    while diff > epsilon
        for j = 1:Nrf
            Vrfj = Vrf(:, [1:j-1, j+1:end]);
            % Compute Cj and Gj as Eq.(13)
            Cj = eye(Nrf-1) + (gamma2/sigma2) * (Vrfj' * F * Vrfj);
            Gj = (gamma2/sigma2) * F - (gamma2/sigma2)^2 * F * Vrfj * pinv(Cj) * Vrfj' * F;
            % Vrf update loop
            for i = 1:N
                eta_ij = 0.0;
                % Sum l != i loop
                for l = setdiff(1:N, i)
                    eta_ij = eta_ij + Gj(i, l) * Vrf(l, j);
                end
                % Value assignment as per (14)
                if eta_ij == 0
                    Vrf(i,j) = 1;
                else
                    Vrf(i,j) = eta_ij / abs(eta_ij);
                end
            end
        % Save the last result
        last_iter_obj = iter_obj;
        % Calculate objective function of (12a)
        iter_obj = log2(det(real(eye(Nrf) + (gamma2/sigma2) * Vrf' * F * Vrf)));
        % Calculate difference of last and current objective function
        diff = abs((iter_obj - last_iter_obj) / iter_obj);
        end
    end
end

function a = stevec_ULA(theta, M)
    % Generates a steering vector for Uniform Linear Array (ULA)
    % theta: rad
    m = 0:M-1;
    a = exp(1i * pi * m * sin(theta))/sqrt(M);
    a = a.';
end

function H = channel(K, N, M, L)
    H = zeros(K, M, N);
    for k = 1:K
        phi_t = (2 * rand(L) - 1)*2*pi;
        phi_r = (2 * rand(L) - 1)*2*pi;
        alphas = (randn(L) + 1j * randn(L))/sqrt(2.0);
        Hk = zeros(M, N);
        for l = 1:L 
            at = stevec_ULA(phi_t(l), N);
            ar = stevec_ULA(phi_r(l), M);
            Hk = Hk + alphas(l) * (ar * at');
        end
        H(k,:,:) = Hk;
    end
    H = H * sqrt(N*M/L); 
end




























