function [x y] = cafFit_uni(N, tp, sr, d_CAF, w)
% [x y] = cafFit_uni(N, tp, sr, d_CAF, w): Cross AF synthesis by
% fitting a desired CAF. The generated x has unit-modulus.
%
%   N: number of sub-pulses.
%   tp: time duration of each sub-pulse
%   sr: samples per sub-pulse
%   d_CAF: desired CAF. (2*Nf-1)-by-(2*Ns-1) where Ns=N*sr and
%   Nf=ceil(N*sr/2)
%   w: weights. (2*Nf-1)-by-(2*Ns-1).
%
%   x: N-by-1
%   y: N-by-1

T = N * tp;
t_grid_size = tp / sr;
f_grid_size = (1/T) / sr;

Ns = N * sr; % no. of samples for the waveform
Nf = ceil(N/2 * sr); % i.e., 1/(2tp) / f_grid_size

% check the dimension of w
if (size(w,1) ~= (2*Nf-1)) || (size(w,2) ~= (2*Ns-1))
    error('the dimension of w is incorrect');
end
% check the dimension of d_CAF
if (size(w,1) ~= (2*Nf-1)) || (size(w,2) ~= (2*Ns-1))
    error('the dimension of d_CAF is incorrect');
end

% grid for time delay
t_grid = (-Ns+1 : Ns-1)' * t_grid_size;

% grid for Doppler frequency
f_grid = (-Nf+1 : Nf-1)' * f_grid_size;

% index for the region of interest specified by w
% the size of w is length(f_grid)-by-length(t_grid)
[p_index k_index] = find(w > 0);

% check if the weighting is all one for any (tau,f)
w_not_one = logical(norm(w - ones(2*Nf-1, 2*Ns-1)));
if w_not_one
    % remove the grid points within the mainlobe
    f_mainlobe = (abs(f_grid(p_index)) <= 1/T); % length(p_index)-by-1
    t_mainlobe = (abs(t_grid(k_index)) <= tp); % length(k_index)-by-1
    p_index(f_mainlobe & t_mainlobe) = [];
    k_index(f_mainlobe & t_mainlobe) = [];
    % add the center point
    p_index = [Nf; p_index];
    k_index = [Ns; k_index];
end

% construct the K matrix
disp('constructing K...'); drawnow update;
N_grid = length(k_index); % same as length(p_index)
K_set = zeros(N, N, N_grid);
for n = 1:N_grid
    disp([num2str(N_grid) ' points: ' num2str(n)]); drawnow update;
    tau = t_grid(k_index(n));
    f = f_grid(p_index(n));
    K_set(:,:,n) = K_tau_f(N, tp, tau, f);
end
% % below is the brute-force calculating of K
% ts = tp / sr; % sampling interval
% for m = 1:N
%     for n = 1:N
%         disp([m n]); drawnow update;
%         for l = 1:N_grid
%             tau = t_grid(k_index(l));
%             f = f_grid(p_index(l));
%             t = ((-N*tp) : ts : (2*N*tp))';
%             pm = 1/sqrt(tp) * pulse((t - (m-1) * tp) / tp, 'rect');
%             pn = 1/sqrt(tp) * pulse((t - (n-1) * tp + tau) / tp, 'rect');
%             K_set(m, n, l) = ts * sum(pm .* conj(pn) .* exp(1i * 2*pi * f * t));
%         end
%     end
% end

% prepare for iteration
disp('start iteration...'); drawnow update;
x = exp(1i * 2*pi * rand(N,1));
y = exp(1i * 2*pi * rand(N,1));
x_pre = zeros(N, 1);
y_pre = zeros(N, 1);
count = 1;
iter_diff = norm(x - x_pre)^2 + norm(y - y_pre)^2;
num_iter = 200;
criterion = zeros(num_iter, 1);

phi = zeros(N_grid, 1);
while (count <= num_iter && iter_diff >= 1e-3)
    disp([num2str(count) ': ' num2str(iter_diff)]);
    disp(['PAR = ' num2str(par(x))]);
    x_pre = x;
    y_pre = y;
    % w.r.t. phi
    for l = 1:N_grid
        phi(l) = angle(y' * K_set(:,:,l) * x);
    end
    % w.r.t. x
    B = zeros(N, N);
    D1 = zeros(N, N);
    for l = 1:N_grid
        p = p_index(l);
        k = k_index(l);
        B = B + w(p,k) * d_CAF(p,k) * exp(1i * phi(l)) * K_set(:,:,l)';
        if w_not_one
            D1 = D1 + w(p,k) * K_set(:,:,l)' * (y * y') * K_set(:,:,l);
        end
    end
    B = B / (N*sr*sr);
    if w_not_one
        D1 = D1 / (N*sr*sr);
    else
        D1 = (y'*y) * eye(N);
    end
    x = D1 \ (B * y); % inv(D1) * B * y
    x = exp(1i * phase(x));
    % w.r.t. y
    if w_not_one
        D2 = zeros(N, N);
        for l = 1:N_grid
            p = p_index(l);
            k = k_index(l);
            D2 = D2 + w(p,k) * K_set(:,:,l) * (x * x') * K_set(:,:,l)';
        end
        D2 = D2 / (N*sr*sr);
    else
        D2 = (x'*x) * eye(N);
    end
    y = D2 \ (B' * x);
    % minimization criterion
    b = B' * x;
    criterion(count) = sum(sum(w .* (abs(d_CAF)).^2)) / (N*sr^2) - ...
        real(b' * (D2 \ b)); % -b' * inv(D2) * b
    count = count + 1;
    iter_diff = norm(x - x_pre)^2 + norm(y - y_pre)^2;
end

% display the criterion vs. iteration
figure;
plot(1:(count-1), criterion(1:count-1), 'rx-');
xlabel('Iteration');
ylabel('Criterion');
myboldify;

% normalize x and y
coef = sqrt(N) / norm(x);
x = x * coef;
y = y / coef;