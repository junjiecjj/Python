%% an example to show Dinkelbach does not converge in sum-of-ratio
x = 30;
num_iter = 30;
ratio = nan(num_iter,1);
for iter = 1:num_iter
    y = x / (x^2+1);
    y0 = 5*x / (x+2);
    ratio(iter) = y + y0;
    cvx_begin
        variable x
        maximize( x - y*(x^2+1) + 5*x - y0*(x+2))
        subject to
            x >= 0
    cvx_end
end

plot(0:num_iter-1,ratio,'-o')