%% superimposed 50 PeCAN sequences of length 256
N = 256;
num = 200;
r = zeros(2*N-1, num);
ISL = zeros(1, num);
X = zeros(N, num);
X0 = zeros(N, num);
for k = 1:num
    disp(['k = ' num2str(k)]); drawnow;
    X0(:,k) = exp(1i * 2*pi * rand(N,1));
    X(:,k) = pecansiso(N, X0(:,k));
    [r(:,k) ISL(k)] = perplotsiso(X(:,k));
end
[tmp index] = sort(ISL);
num = 50;
r = r(:, index(1:num));
X = X(:, index(1:num));
X0 = X0(:, index(1:num));
figure;
for k = 1:num
    plot(-(N-1):(N-1), 20*log10(abs(r(:,k))/N)); hold on;
end
hold off;
xlabel('k'); ylabel('$|\tilde{r}(k)|/N$ (dB)', 'Interpreter', 'LaTex');
axis([-N+1 N-1 -250 0]);
myboldify;
myresize('PeCAN1');
figure;
for k = 1:num
    r_init = perplotsiso(X0(:,k));
    plot(-(N-1):(N-1), 20*log10(abs(r_init)/N)); hold on;
end
hold off;
xlabel('k'); ylabel('$|\tilde{r}(k)|/N$ (dB)', 'Interpreter', 'LaTex');
axis([-N+1 N-1 -250 0]);
myboldify;
myresize('PeCAN1_init');
% compute cross correlation
all_r = zeros(num * (num-1) / 2 * (2*N-1), 1);
count = 0;
for k1 = 1:num
    for k2 = (k1+1):num
        count = count + 1;
        disp(['k1 = ' num2str(k1) ', k2 = ' num2str(k2)]);
        all_r((count-1)*(2*N-1)+1 : count*(2*N-1)) = ...
            perplotcross(X(:,k1), X(:,k2));
    end
end
all_r = abs(all_r);
r_min = min(all_r/N);
r_max = max(all_r/N);
r_med = median(all_r/N);
disp(['min corr = ' num2str(r_min)]);
disp(['max_corr = ' num2str(r_max)]);
disp(['median_corr = ' num2str(r_med)]);
% save pecan.mat