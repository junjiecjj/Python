num_iter = 50;
max_ratio = 1/2;

%% Quadratic
y = nan(num_iter+1,1);
y(1) = 0.1;
for t = 1:num_iter
    y(t+1) = (2*y(t))^(-1/3) / ((2*y(t))^(-4/3)+1);
end
x = (2*y).^(-2/3);
ratio = x./(x.^2+1);

%% Dinkelbach
% x(1) equals to that of Quadratic
for t = 1:num_iter
    y(t) = x(t)/(x(t)^2+1);
    x(t+1) = 1/2/y(t);
end

figure; hold on
plot(0:num_iter,max_ratio-y,'r'); % Dinkelbach
plot(0:num_iter,max_ratio-ratio); 
