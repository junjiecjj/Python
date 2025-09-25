clear
step = 0.01;
p = zeros(2,1);

H = [1,0.1;0.09,0.87]; % channel of legimate users
h = [0.5,0.11;0.13,0.39];    %  illegimate users
sigma_E = ones(2,1);     % noise power at eavesdroppers (Eve)  -80dBm
sigma_B = 1e-1*ones(2,1);     % noise power at users with eavesdroppers (Bob)  -90dBm
opt = Compute_obj(sigma_B,sigma_E,10*ones(2,1),H,h);
opt_p = zeros(2,1);
tic
for i = 0:1000
    p(1) = i*step;
    for j = 0:1000
        p(2) = j*step;
        cur_val =  Compute_obj(sigma_B,sigma_E,p,H,h);
        if cur_val > opt
            opt = cur_val;
            opt_p = p;
        end
    end
end
toc