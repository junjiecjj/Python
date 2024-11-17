function [res, input_par, output_par] = gamp_bgm(A, y, gamp_par, input_par, output_par)

    % use MMSE formulation of GAMP with scalar variance
    % the prior distribution of the input channel is Bernoulli-Gaussian mixture
    % the prior distribution of the output channel is multi-bit quantization noise model
    M   = size(A,1);    % the dimensionality of the measurement y
    N   = size(A,2);    % the dimensionality of the signal x
    
    A_sq_sum = gamp_par.A_sq_sum;  % compute the squared frobenius norm of A
    
    % set GAMP parameters
    max_pe_ite = gamp_par.max_pe_ite;   % maximum number of parameter estimation iterations
    cvg_thd = gamp_par.cvg_thd; % the convergence threshold
    
    x_hat   = gamp_par.x_hat;   % estimated signal
    tau_x   = gamp_par.tau_x;   % signal variance
    s_hat   = gamp_par.s_hat;   % dummy variable from output function
    
    verbose = gamp_par.verbose;	% whether to output convergence values in every iteration
    
    % set input distribution parameters
    lambda  = input_par.lambda; % Bernoulli parameter
    theta   = input_par.theta;  % the means of the Gaussian mixtures
    phi     = input_par.phi;    % the variances of the Gaussian mixtures
    omega   = input_par.omega;  % the weights of the Gaussian mixtures
    num_c   = input_par.num_c;  % the number of Gaussian mixtures
    
    % set output distribution parameters
    tau_w   = output_par.tau_w; % the white-Gaussian noise variance
    
    % set quantization threshold
    quant_thd = output_par.quant_thd;
    
    % process y into lower and upper theshold based on quant_thd
    % note that y is supposed to go from 2 to 2^K+1 continuously
    y_lower = zeros(length(y),1);
    y_upper = zeros(length(y),1);
    for (i=1:length(y))
        y_lower(i) = quant_thd(y(i)-1);
        y_upper(i) = quant_thd(y(i));
    end
    
    % set the lower bound and upper bound to -inf and +inf
    % maybe have an option flag to determine whether to do this or not
    y_lower(1) = -Inf;
    y_upper(length(y_upper)) = Inf;


    for (ite_pe = 1:max_pe_ite)

        x_hat_pe_pre = x_hat;
        
        % output linear step
        tau_p = A_sq_sum/M * tau_x;
        p_hat = A * x_hat - tau_p * s_hat;
            
        % output nonlinear step
        [s_hat, tau_s] = output_function(p_hat, tau_p, y_lower, y_upper, tau_w);

        if (tau_s<eps)
            tau_s = eps;
        end

        % input linear step
        tau_r = 1 / (tau_s * A_sq_sum / N);
        r_hat = x_hat + tau_r * (s_hat' * A)';

        % input nonlinear step
        [x_hat, tau_x] = input_function(r_hat, tau_r, lambda, omega, theta, phi);
        
        cvg_val_pe = norm(x_hat/sum(abs(x_hat))-x_hat_pe_pre/sum(abs(x_hat_pe_pre))) / norm(x_hat/sum(abs(x_hat)));
        
        if (verbose==1)
        	fprintf('Iteration %d: %f\n', ite_pe, cvg_val_pe);
        end

        if (cvg_val_pe<cvg_thd) 
            fprintf('Convergence reached\n');
            %fprintf('Lambda: %f\n', lambda);
            %fprintf('Omega: ');
            %for (i=1:num_c)
            %    fprintf('%f ', omega(i))
            %end
            %fprintf('\n')

            %fprintf('Theta: ');
            %for (i=1:num_c)
            %    fprintf('%f ', theta(i))
            %end
            %fprintf('\n')

            %fprintf('Phi: ')
            %for (i=1:num_c)
            %    fprintf('%f ', phi(i))
            %end
            %fprintf('\n')

            %fprintf('Tau: %f\n', tau_w);
            
            break;
        end
    end
    
    % save the recovery results
    res.x_hat = x_hat;  % the recovered signal
    res.tau_x = tau_x;
    res.s_hat = s_hat;
    res.tau_s = tau_s;
    res.p_hat = p_hat;
    res.tau_p = tau_p;
    res.r_hat = r_hat;
    res.tau_s = tau_r;
    
    % update input distribution parameters
    input_par.lambda = lambda; % Bernoulli parameter
    input_par.theta = theta;  % the means of the Gaussian mixtures
    input_par.phi   = phi;    % the variances of the Gaussian mixtures
    input_par.omega = omega;  % the weights of the Gaussian mixtures
    
    % update output distribution parameters
    output_par.tau_w = tau_w; % the white-Gaussian noise variance

end

function [x_hat, tau_x] = input_function(r_hat, tau_r, lambda, omega, theta, phi)
    
    dim_smp=length(r_hat);
    num_cluster=length(omega);

    cluster_block_exp_mat = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        cluster_block_exp_mat(:,i) = -0.5 * ( (theta(i)-r_hat).^2 ./ (phi(i)+tau_r) );
    end

    cluster_block_exp_max = zeros(dim_smp, 1);
    for (i=1:dim_smp)
        cluster_block_exp_max(i) = max(cluster_block_exp_mat(i,:));
    end

    block_mat = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        block_mat(:,i) = omega(i) * sqrt(tau_r/(phi(i)+tau_r)) * exp( -cluster_block_exp_max -0.5 * (theta(i)-r_hat).^2 / (phi(i)+tau_r) );
    end

    % compute x_hat
    block_mat_nmr_x = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x(:,i) = block_mat_nmr_x(:,i) .* (theta(i) * tau_r + r_hat * phi(i)) ./ (phi(i) + tau_r);
    end

    nmr_x = sum(block_mat_nmr_x, 2);
    dnm_x = sum(block_mat, 2) + ((1-lambda)/lambda) * exp(-cluster_block_exp_max-0.5*r_hat.^2/tau_r);

    x_hat = nmr_x ./ (dnm_x);

    % compute tau_x
    block_mat_nmr_x_sq = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x_sq(:,i) = block_mat_nmr_x_sq(:,i) .* (  phi(i)*tau_r/(phi(i) + tau_r) + ( (theta(i)*tau_r + r_hat*phi(i)) / (phi(i) + tau_r) ).^2  );
    end

    nmr_x_sq = sum(block_mat_nmr_x_sq, 2);
    dnm_x_sq = dnm_x;

    tau_x = mean(nmr_x_sq ./ dnm_x_sq - x_hat.^2);  % this is nonnegative in theory
    tau_x = max(tau_x, 1e-12);  % just in case

end

function [s_hat, tau_s] = output_function(p_hat, tau_p, y_lower, y_upper, tau_w)
   
    p_bar = p_hat/sqrt(tau_w+tau_p);
    y_lower_bar = y_lower/sqrt(tau_w+tau_p);
    y_upper_bar = y_upper/sqrt(tau_w+tau_p);
    y_upper_bar_minus_p_bar = y_upper_bar-p_bar;
    y_lower_bar_minus_p_bar = y_lower_bar-p_bar;
    exp_y_upper_bar_minus_p_bar = exp(-0.5*(y_upper_bar_minus_p_bar).^2);
    exp_y_lower_bar_minus_p_bar = exp(-0.5*(y_lower_bar_minus_p_bar).^2);
    mut_exp_y_upper_bar_minus_p_bar = (y_upper_bar_minus_p_bar).*exp_y_upper_bar_minus_p_bar;
    mut_exp_y_upper_bar_minus_p_bar((abs(y_upper_bar_minus_p_bar)==Inf)&(exp_y_upper_bar_minus_p_bar==0))=0;
    mut_exp_y_upper_bar_minus_p_bar((y_upper_bar_minus_p_bar==0)&(abs(exp_y_upper_bar_minus_p_bar)==Inf))=0;
    mut_exp_y_lower_bar_minus_p_bar = (y_lower_bar_minus_p_bar).*exp_y_lower_bar_minus_p_bar;
    mut_exp_y_lower_bar_minus_p_bar((abs(y_lower_bar_minus_p_bar)==Inf)&(exp_y_lower_bar_minus_p_bar==0))=0;
    mut_exp_y_lower_bar_minus_p_bar((y_lower_bar_minus_p_bar==0)&(abs(exp_y_lower_bar_minus_p_bar)==Inf))=0;

    
    %exp_diff_block = exp(-0.5*(y_upper_bar_minus_p_bar).^2)-exp(-0.5*(y_lower_bar_minus_p_bar).^2);
    %exp_diff_block_mod = (y_upper_bar_minus_p_bar).*exp(-0.5*(y_upper_bar_minus_p_bar).^2)-(y_lower_bar_minus_p_bar).*exp(-0.5*(y_lower_bar_minus_p_bar).^2);

    exp_diff_block = exp_y_upper_bar_minus_p_bar - exp_y_lower_bar_minus_p_bar;
    exp_diff_block_mod = mut_exp_y_upper_bar_minus_p_bar - mut_exp_y_lower_bar_minus_p_bar;

    PI_0 = 0.5*(erf(sqrt(0.5)*(y_upper_bar_minus_p_bar))-erf(sqrt(0.5)*(y_lower_bar_minus_p_bar))); % to improve the precision
    PI_0(abs(PI_0)<eps) = eps;
    
    PI_1 = p_hat.*PI_0 -tau_p/sqrt(2*pi*(tau_w+tau_p))*exp_diff_block;
    
    PI_2 = (p_hat.^2+tau_p).*PI_0 - (1/sqrt(2*pi))/(tau_w+tau_p)*(tau_p^2).*exp_diff_block_mod - 2/sqrt(2*pi)*tau_p*p_bar.*exp_diff_block;
    
    z_hat = PI_1./PI_0;
    z_hat_sq = PI_2./PI_0;
    
    tau_z = z_hat_sq-z_hat.^2;
   
    s_hat = 1/tau_p*(z_hat-p_hat);
    tau_s = 1/tau_p*(1-tau_z/tau_p);
    tau_s = mean(tau_s);    % should I use mean here???

end
