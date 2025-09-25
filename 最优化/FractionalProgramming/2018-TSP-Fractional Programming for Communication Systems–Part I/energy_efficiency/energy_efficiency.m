clc; clear

numIter = 50; % num of iter
utility_dinkelbach = zeros(numIter,1);
utility_proposed = zeros(numIter,1);

F = 1;
max_p = 10^((21-30)/10)*F;
noise =  10^((-100-30)/10)*F;
po = 10^((5-30)/10)*F;
chn = 1e-12;

num_iter = 20;

%% plot the function
p = 0:max_p/100:max_p;
ee = log2(1+chn*p/noise)./(p+po);
plot(p,ee);

%% compare convergence
ee_dinkelbach = nan(num_iter,1); % energy efficiency by Dinkelbach
ee_quadratic = nan(num_iter,1); % energy efficiency by Quadratic

% Dinkelbach
p = max_p;
for iter = 1:num_iter
    y = log(1+chn*p/noise)/(p+po);
    ee_dinkelbach(iter) = y/log(2);
    
    cvx_begin
        variable p
        maximize( log(1+chn*p/noise) - y*(p+po) )
        subject to
            p >= 0
            p <= max_p - po
    cvx_end
end

% quadratic
p = max_p;
for iter = 1:num_iter
    y = sqrt(log(1+chn*p/noise))/(p+po);
    ee_quadratic(iter) = log2(1+chn*p/noise)/(p+po);
    
    cvx_begin
        variable p
        maximize( 2*y*sqrt(log(1+chn*p/noise)) - y^2*(p+po) )
        subject to
            p >= 0
            p <= max_p - po
    cvx_end
end

figure; hold on
plot(0:num_iter-1, ee_dinkelbach,'r-o')
plot(0:num_iter-1, ee_quadratic,'b-*')

% figure; hold on
% plot(1:num_iter, max(ee_dinkelbach)-ee_dinkelbach,'r')
% plot(1:num_iter, max(ee_quadratic)-ee_quadratic)
% 
% figure; hold on
% gap_dinkelbach = max(ee_dinkelbach)-ee_dinkelbach;
% gap_quadratic = max(ee_quadratic)-ee_quadratic;
% plot(0:5, gap_dinkelbach(1:6),'r')
% plot(0:9, gap_quadratic(1:10))
