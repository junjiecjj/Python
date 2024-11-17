sigma = 2;			% the oversampling ratio, i.e. beta
rho = 0.1;			% the sparsity ratio
N = 10000;       	% the signal length
M = sigma*N;      	% the number of measurements
S = ceil(rho*N);  	% the number of nonzero entries


mc_num = 200000; 	% the number of Monte Carlo simulates to calculate the expectations in the paper

num_c_true = 1; 	% the true number of Gaussian mixtures
meas_snr_val = 30;	% pre-quantization SNR (dB)
gaussian_num = 1;   % the standard deviation of Gaussian component in BGM prior
bit_num = 1;		% the quantization bit number

LN_num = 2;						% the number of computing threads
LN = maxNumCompThreads(LN_num);	% set the largest number of computing threads

max_pe_ite = 20;		% the maximum number of iterations for AMP-PE and AMP-AWGN
max_pe_inner_ite = 20;	% the maximum number of inner iterations to estimate the parameters
cvg_thd = 1e-6; 		% convergence threshold
kappa = 1;      		% damping rate for parameter estimation
verbose = 0;			% "1" - output convergence values in every iteration; "0" - do not output convergence values in every iteration

% set the quantization thresholds and step size properly
% they need to be adjusted for different signals
quant_thd_min = -1;										% the minimum quantization threshold
quant_thd_max = 1;										% the maximum quantization threshold
% just make sure the precision is high enough when rho = 0.1
if (rho==0.1)
    quant_thd_min = -1.3;
    quant_thd_max = 1.3;
end 
if (rho==0.5)
    quant_thd_min=-2.7;
    quant_thd_max=2.7;
end 
if (rho==1)
    quant_thd_min=-4;
    quant_thd_max=4;
end 


quant_step = (quant_thd_max-quant_thd_min)/(2^bit_num);	% quantizatin step size

% compute the quantization thresholds
quant_thd = quant_thd_min;
for (i=(1:(2^bit_num -1)))
    quant_thd = [quant_thd quant_thd_min+i*quant_step];
end
quant_thd = [quant_thd quant_thd_max];

mse_seq_mat = [];	% the matrix holding all the signal to noise ration values

% run 10 random trials here
for (trial_num = 1:10)
    fprintf('Trial %d\n', trial_num)

    rng(trial_num); % set random seed
    snr_seq = [];	% the vector holding the snr values at each random trial

    %%%%%%%%%%%%%%%%%%%
    %% create signal %%
    %%%%%%%%%%%%%%%%%%%

    % true Gaussian mixture weights
    omega_true = rand(num_c_true,1);
    omega_true = omega_true/sum(omega_true);

    nonzeroW = [];      % nonzero entries
    theta_true = [];    % true Gaussian mixtures means
    phi_true = [];      % true Gaussian mixtures variances
    for (i=1:num_c_true)
        theta_tmp = 0;  % make sure the mean is 0 for this particular 1bit CS 
        theta_true = [theta_true; theta_tmp];
        phi_tmp = gaussian_num;	% fix the variance to make sure the noise level is fixed
        phi_true = [phi_true; phi_tmp];
        S_tmp = round(S*omega_true(i));
        nonzeroW = [nonzeroW; normrnd(theta_tmp, gaussian_num, S_tmp, 1)];  % the nonzero entries
    end

    % double check since we rounded the number  before
    if (length(nonzeroW)>N)
        nonzeroW = nonzeroW(1:N);
    end

    ind = randperm(N);
    indice = ind(1:length(nonzeroW));
    x = zeros(N, 1);
    x(indice) = nonzeroW;   % true signal

    % the measurement matrix A, this is different from the experimental section
    A = randn(M,N) * 1/sqrt(M);

	% needed for scalar-version AMP computation    
    A_sq_sum = norm(A, 'fro')^2;  % compute the squared frobenius norm of A

    y_noiseless = A*x;    % noiseless linear measurements
    
    noise_ori = normrnd(0, 1, size(y_noiseless));
    mut_factor = sqrt(sum(y_noiseless.^2)/sum(noise_ori.^2) / 10^(meas_snr_val/10));

    y_ori = y_noiseless + mut_factor * noise_ori;   % noisy measurements


    % convert "y_ori" to quantized symbols "y" in the set {2,3,4,5,6,...}
    % "y_quant" is the "low resolution" version of the "y_ori"
    
	y = zeros(size(y_ori));
    y_quant = zeros(size(y_ori));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% note that the quantized symbols start from "2" !!! %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for (i=1:length(y_ori))
        for (j=2:length(quant_thd))
            if ( (y_ori(i)>=quant_thd(j-1))&&(y_ori(i)<quant_thd(j)) )
                y(i) = j;
                y_quant(i) = 0.5*(quant_thd(j-1)+quant_thd(j));
                break;
            end
        end
    end

    % just in case the measurements are beyond the quantization range
    for (i=1:length(y_ori))
        if (y_ori(i)<quant_thd(1))
            y(i) = 2;
            y_quant(i) = 0.5*(quant_thd(1)+quant_thd(2));
        end
        if (y_ori(i)>=quant_thd(length(quant_thd)))
            y(i) = length(quant_thd);
            y_quant(i) = 0.5*(quant_thd(length(quant_thd)-1)+quant_thd(length(quant_thd)));
        end
    end
    

    %%%%%%%%%%%%%%%%%%%%
    %% initialization %%
    %%%%%%%%%%%%%%%%%%%%

    % use the l2-min solution to get an estimate of the distribution parameters
    % if you have some vague notion about the distribution parameters, you can directly set the distribution parameters
    x_hat = zeros(N,1);
    d_cg = zeros(N,1);
    d_cg = -A'*(A*x_hat-y_quant);
    r_cg = d_cg;

	% use conjungate gradient descent to compute the least square solution
    max_cg_ite = 20; 

    for (ite=1:max_cg_ite)
        a_cg_n = sum(r_cg.*r_cg);
        d_cg_tmp = A*d_cg;
        a_cg_d = sum(d_cg_tmp.*d_cg_tmp);
        a_cg = a_cg_n / a_cg_d;
        x_hat_pre = x_hat;
        x_hat = x_hat + a_cg*d_cg;
        cvg_cg_val = norm(x_hat-x_hat_pre,'fro')/norm(x_hat,'fro');
        if (cvg_cg_val<cvg_thd)
            break;
        end

        r_cg_new = r_cg - a_cg*(A'*(A*d_cg));

        b_cg = sum(r_cg_new.*r_cg_new)/sum(r_cg.*r_cg);
        d_cg = r_cg_new + b_cg*d_cg;

        r_cg = r_cg_new;
    end

	% initialize the AMP algorithm
    tau_x = var(x_hat)+1e-12;	% initialize scalar signal variance
    s_hat = zeros(M,1); 		% initialize s_hat with all zeros

    % initialize input distribution parameters
    lambda = 0.1;			% the sparsity ratio
	num_c = 1;				% the number of Gaussian mixture component

    % initialize Gaussian mixtures parameters
    theta=zeros(num_c, 1);  % Gaussian mean
    phi=ones(num_c, 1);     % Gaussian variance
    omega=zeros(num_c,1);   % Gaussian mixture weights

    x_hat_nz = x_hat(x_hat~=0);
    idx_seq = kmeans(x_hat_nz, num_c);
    for (i=1:num_c)
        x_hat_tmp = x_hat_nz(idx_seq==i);
        if (length(x_hat_tmp)>0)
            theta(i) = mean(x_hat_tmp);
            phi(i) = var(x_hat_tmp)+1e-12;  % avoid zero variance
            omega(i) = length(x_hat_tmp)/length(x_hat_nz);
        else 
            theta(i) = 0;
            phi(i) = 1e-12; % avoid zero variance
            omega(i) = 0;
        end
    end

    % initialize white Gaussian noise variance
    tau_w = 1e-6;
    
    % initialize with all zero vector
    x_hat = zeros(N,1);

    % set GAMP parameters
    gamp_par.A_sq_sum	= A_sq_sum;			% the Frobenius norm of the measurement matrix
    gamp_par.max_pe_ite    = max_pe_ite;	% maximum number of iterations for AMP
    gamp_par.max_pe_inner_ite = max_pe_inner_ite;	% maximum number of inner iterations for parameter estimation
    gamp_par.cvg_thd    = cvg_thd; 			% the convergence threshold
    gamp_par.kappa      = kappa;   			% learning rate
    gamp_par.verbose	= verbose;			% "0" or "1", decides whether to output convergence values in each iteration

    gamp_par.x_hat      = x_hat; 		  			% initialize with all zero vector
    gamp_par.tau_x      = var(x);					% true signal variance here since we are doing state evolution
    gamp_par.s_hat      = s_hat;   					% dummy variable from output function
    
    gamp_par.x_true		= x;		% true signal used to calculate MSE

    % set input distribution parameters
    input_par.lambda    = lambda;	% the sparsity ratio
    input_par.theta     = theta;	% the means of the Gaussian mixtures
    input_par.phi       = phi;    	% the variances of the Gaussian mixtures
    input_par.omega     = omega;  	% the weights of the Gaussian mixtures
    input_par.num_c     = num_c;  	% the number of Gaussian mixtures

    % set output distribution parameters
    output_par.tau_w    = tau_w; 		% the white-Gaussian noise variance
    output_par.quant_thd = quant_thd; 	% quantization threshold
    
    
    % recovery the signal from quantized measurements using different approaches

    %%%%%%%%%%%%%%%%%%%%%
    %% AMP-PE recovery %%
    %%%%%%%%%%%%%%%%%%%%%

    [res, input_par_new, output_par_new] = gamp_bgm_multi_bit_scalar_SE_verify(A, y, gamp_par, input_par, output_par);

    mse_seq_tmp = zeros(1,max_pe_ite+1);
    mse_seq_tmp(1:length(res.mse_seq)) = res.mse_seq;
    mse_seq_mat = [mse_seq_mat; mse_seq_tmp];
    
end


    
%%%%%%%%%%%%%%%%%%%%%
%% State Evolution %%
%%%%%%%%%%%%%%%%%%%%%

% use the initial parameters from the last random trial to do the SE

clear gamp_par input_par output_par;

% set GAMP parameters
gamp_par.max_pe_ite    = max_pe_ite;	% maximum number of iterations for AMP
gamp_par.max_pe_inner_ite = max_pe_inner_ite;	% maximum number of inner iterations for parameter estimation
gamp_par.cvg_thd    = cvg_thd; 			% the convergence threshold
gamp_par.kappa      = kappa;   			% learning rate
gamp_par.verbose	= verbose;			% "0" or "1", decides whether to output convergence values in each iteration

gamp_par.tau_x		= var(x);		% true signal variance here since we are doing state evolution
gamp_par.mc_num		= mc_num;		% the number of Monte Carlo simulations used to calculate the expectations
gamp_par.beta		= M/N;			% the oversampling ratio

% set input distribution parameters
input_par.lambda    = lambda;	% the sparsity ratio
input_par.theta     = theta;	% the means of the Gaussian mixtures
input_par.phi       = phi;    	% the variances of the Gaussian mixtures
input_par.omega     = omega;  	% the weights of the Gaussian mixtures
input_par.num_c     = num_c;  	% the number of Gaussian mixtures

% set true input distribution parameters
input_par.lambda_true = rho;
input_par.theta_true = theta_true;
input_par.phi_true = phi_true;
input_par.omega_true = omega_true;
input_par.num_c_true = num_c_true;

% set output distribution parameters
output_par.tau_w    = tau_w; 		% the white-Gaussian noise variance

% set true output distribution parameters
output_par.tau_w_true    = mut_factor^2; 		% the white-Gaussian noise variance

output_par.quant_thd = quant_thd; 	% quantization threshold

[res, input_par_new, output_par_new] = gamp_bgm_multi_bit_scalar_SE(gamp_par, input_par, output_par);

% res.tau_x_seq is the predicted MSE values through the iterations


% plot the predicted MSE values from state evolution and the actual MSE values from the random trials
figure;
ite_seq = 1:(max_pe_ite+1);
for (i=1:size(mse_seq_mat,1))
plot(ite_seq, mse_seq_mat(i,:))
hold on
end
plot(ite_seq, res.tau_x_seq, '--k', 'LineWidth', 2)
set(gca, 'YScale', 'log')
ylabel('MSE (log scale)')
xlabel('Iteration')
title('1-bit CS: M/N=2, E/N=10%, pre-QNT SNR=30dB')
legend('Random trial 1','Random trial 2','Random trial 3','Random trial 4','Random trial 5','Random trial 6','Random trial 7','Random trial 8','Random trial 9','Random trial 10','State Evolution')
hold off

