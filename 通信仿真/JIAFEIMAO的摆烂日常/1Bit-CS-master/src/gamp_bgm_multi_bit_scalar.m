function [res, input_par, output_par] = gamp_bgm(A, y, gamp_par, input_par, output_par)

    % use MMSE formulation of GAMP with scalar variance
    % the prior distribution of the input channel is Bernoulli-Gaussian mixture
    % the prior distribution of the output channel is multi-bit quantization noise model
    M   = size(A,1);    % the dimensionality of the measurement y
    N   = size(A,2);    % the dimensionality of the signal x
    
    A_sq_sum = gamp_par.A_sq_sum;  % compute the squared frobenius norm of A
    
    % set GAMP parameters
    max_pe_ite = gamp_par.max_pe_ite;   % maximum number of iterations for AMP
    max_pe_inner_ite = gamp_par.max_pe_inner_ite;	% maximum number of iterations for parameter estimation
    cvg_thd = gamp_par.cvg_thd; % the convergence threshold
    kappa   = gamp_par.kappa;   % learning rate of parameter estimation
    eta     = gamp_par.eta;     % damping rate of signal recovery
    
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
    gamma   = input_par.gamma;  % the weight of the outlier distribution (a zero-mean Gaussian distribution)
    psi     = input_par.psi;    % the variance of the outlier distribution (a zero-mean Gaussian distribution)
    
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


    tau_p = A_sq_sum/M * tau_x; 
    p_hat = A * x_hat - tau_p * s_hat;

    for (ite_pe = 1:max_pe_ite)

        x_hat_pe_pre = x_hat;
        
        % output nonlinear step
        [s_hat, tau_s] = output_function(p_hat, tau_p, y_lower, y_upper, tau_w);

        if (tau_s<eps)
            tau_s = eps;
        end

        % input linear step
        tau_r = 1 / (tau_s * A_sq_sum / N);
        r_hat = x_hat + tau_r * (s_hat' * A)';

		% parameter estimation
        for (ite_par=1:max_pe_inner_ite)
        	lambda_pre = lambda;
        	omega_pre = omega;
        	theta_pre = theta;
        	phi_pre = phi;
            gamma_pre = gamma;
            psi_pre = psi;
        	
            [lambda, omega, theta, phi, gamma, psi] = input_parameter_est(r_hat, tau_r, lambda, omega, theta, phi, gamma, psi, kappa);
            
            cvg_lambda = max(abs((lambda_pre-lambda)./lambda));
            cvg_omega = max(abs((omega_pre-omega)./omega));
            cvg_theta = max(abs((theta_pre-theta)./theta));
            cvg_phi = max(abs((phi_pre-phi)./phi));
            cvg_gamma = max(abs((gamma_pre-gamma)./gamma));
            cvg_psi = max(abs((psi_pre-psi)./psi));
            
            cvg_all_parameters = max([cvg_lambda cvg_omega cvg_theta cvg_phi cvg_gamma cvg_psi]);
            
            if (cvg_all_parameters<cvg_thd)
            	break;
            end
        end

        % input nonlinear step
        x_hat_pre = x_hat;
        [x_hat, tau_x] = input_function(r_hat, tau_r, lambda, omega, theta, phi, gamma, psi);
        x_hat = x_hat_pre + eta*(x_hat-x_hat_pre);

        % output linear step
        tau_p = A_sq_sum/M * tau_x;
        p_hat = A * x_hat - tau_p * s_hat;
        
        % parameter estimation
        for (ite_par=1:max_pe_inner_ite)
        	tau_w_pre = tau_w;
        	
            tau_w = output_parameter_est(y_lower, y_upper, tau_w, p_hat, tau_p, kappa);
            
            cvg_tau_w = max(abs((tau_w_pre-tau_w)./tau_w));
            
            if (cvg_tau_w<cvg_thd)
            	break;
            end
        end

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
    input_par.lambda = lambda;  % Bernoulli parameter
    input_par.theta = theta;    % the means of the Gaussian mixtures
    input_par.phi   = phi;      % the variances of the Gaussian mixtures
    input_par.omega = omega;    % the weights of the Gaussian mixtures
    input_par.gamma = gamma;    % the weight of the outlier distribution (a zero-mean Gaussian distribution)
    input_par.psi = psi;        % the variance of the outlier distribution (a zero-mean Gaussian distribution)
    
    % update output distribution parameters
    output_par.tau_w = tau_w; % the white-Gaussian noise variance

end

function [x_hat, tau_x] = input_function(r_hat, tau_r, lambda, omega, theta, phi, gamma, psi)
    
    dim_smp=length(r_hat);
    num_cluster=length(omega);

    block_mat = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        block_mat(:,i) = (1-gamma) * omega(i) * sqrt(tau_r/(phi(i)+tau_r)) * exp( -0.5 * (theta(i)-r_hat).^2 / (phi(i)+tau_r) );
    end

    % compute x_hat
    block_mat_nmr_x = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x(:,i) = block_mat_nmr_x(:,i) .* (theta(i) * tau_r + r_hat * phi(i)) / (phi(i) + tau_r);
    end

    block_mat_2 = gamma * sqrt(tau_r/(psi+tau_r)) * exp( -0.5*r_hat.^2/(psi+tau_r) );
    block_mat_nrm_x_2 = block_mat_2 .* (r_hat*psi ./ (psi+tau_r));
    

    nmr_x = sum(block_mat_nmr_x, 2) + block_mat_nrm_x_2;
    dnm_x = sum(block_mat, 2) + block_mat_2 + ((1-lambda)/lambda) * exp(-0.5*r_hat.^2/tau_r);

    x_hat = nmr_x ./ (dnm_x);

    % if dnm_x is zero, set x_hat to r_hat
    dnm_x_zero_idx = (dnm_x==0);
    x_hat(dnm_x_zero_idx) = r_hat(dnm_x_zero_idx);

    % compute tau_x
    block_mat_nmr_x_sq = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x_sq(:,i) = block_mat_nmr_x_sq(:,i) .* (  phi(i)*tau_r/(phi(i) + tau_r) + ( (theta(i)*tau_r + r_hat*phi(i)) / (phi(i) + tau_r) ).^2  );
    end

    block_mat_nmr_x_sq_2 = block_mat_2 .* ( psi*tau_r/(psi+tau_r) + ((r_hat*psi)/(psi+tau_r)).^2 );

    nmr_x_sq = sum(block_mat_nmr_x_sq, 2) + block_mat_nmr_x_sq_2;
    dnm_x_sq = dnm_x;

    tau_x_seq = (nmr_x_sq ./ dnm_x_sq - x_hat.^2);  % this is nonnegative in theory

    % if dnm_x is zero, set tau_x_seq to 0
    tau_x_seq(dnm_x_zero_idx) = 0;

    tau_x = mean(tau_x_seq);
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

function [lambda, omega, theta, phi, gamma, psi] = input_parameter_est(r_hat, tau_r, lambda, omega, theta, phi, gamma, psi, kappa)
   
    lambda_pre = lambda;
    omega_pre = omega;
    theta_pre = theta;
    phi_pre = phi;
    gamma_pre = gamma;
    psi_pre = psi;

    dim_smp=length(r_hat);
    num_cluster=length(omega);


    lambda_tmp_mat_1 = zeros(dim_smp, num_cluster);
    for (i = 1:num_cluster)
        lambda_tmp_mat_1(:,i) = lambda * (1-gamma) * omega(i) * 1/sqrt(tau_r+phi(i)) * exp( -0.5 * (r_hat-      theta(i)).^2 / (tau_r+phi(i)) );
    end

    lambda_tmp_3 = lambda * gamma * 1/sqrt(tau_r+psi) * exp( -0.5*r_hat.^2 /(tau_r+psi));


    % compute lambda
    lambda_tmp_1 = sum(lambda_tmp_mat_1, 2);
    lambda_tmp_2 = (1-lambda) * 1 / sqrt(tau_r) .* exp(-0.5 * r_hat.^2 / tau_r);

    lambda_tmp_sum = lambda_tmp_1 + lambda_tmp_2 + lambda_tmp_3;
    lambda_tmp_sum_zero_idx = (lambda_tmp_sum==0);    % leave the outliers to the first Gaussian component

    % this is where the outlier model comes into play when lambda_tmp_1+lambda_tmp_2 is zero
    lambda_block_2 = lambda_tmp_2 ./ lambda_tmp_sum;
    lambda_block_2(lambda_tmp_sum_zero_idx) = 0;

    lambda_block_1 = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        lambda_block_1(:,i) = lambda_tmp_mat_1(:,i) ./ lambda_tmp_sum;
        lambda_block_1(lambda_tmp_sum_zero_idx,i) = 0;
    end

    lambda_block_3 = lambda_tmp_3 ./ lambda_tmp_sum;
    lambda_block_3(lambda_tmp_sum_zero_idx) = 1;


    lambda_new = (sum(sum(lambda_block_1)) + sum(lambda_block_3)) / (sum(lambda_block_2) + sum(sum(lambda_block_1)) + sum(lambda_block_3));
    lambda = lambda + kappa * (lambda_new - lambda);

    
    % compute omega
    omega_new = sum(lambda_block_1);
    omega_new = omega_new';
    omega_new = omega_new / sum(omega_new);

    omega = omega + kappa * (omega_new - omega);


    % compute gamma
    gamma_new = sum(lambda_block_3) / (sum(sum(lambda_block_1)) + sum(lambda_block_3));

    %gamma_new
    gamma = gamma + kappa * (gamma_new - gamma);
    
    % compute theta
    % this is where the outlier model comes into play when lambda_tmp_mat_sum+lambda_tmp_2 is zero
    theta_tmp_mat = lambda_block_1;
    theta_tmp_mat_sum_1 = sum(theta_tmp_mat);
    zero_idx=(theta_tmp_mat_sum_1==0);  % find the guassian mixtures with zero weight
    
    theta_tmp_mat_1 = theta_tmp_mat;
    theta_tmp_mat_2 = theta_tmp_mat;
    for (i=1:num_cluster)
        theta_tmp_mat_1(:,i) = theta_tmp_mat_1(:,i) .* ( r_hat ./ (phi(i)+tau_r) );
        theta_tmp_mat_2(:,i) = theta_tmp_mat_2(:,i) ./ (phi(i)+tau_r);
    end
    theta_new = sum(theta_tmp_mat_1) ./ sum(theta_tmp_mat_2);   % to avoid division by 0
    theta_new(zero_idx) = theta(zero_idx);  % keep the gaussian mixtures with zero weight fixed

    theta_new_inf_idx = isinf(theta_new);   % guard against the case of dision by 0
    theta_new(theta_new_inf_idx) = theta(theta_new_inf_idx); % keep the gaussian mixtures with infinity theta fixed just in case

    theta_new = theta_new';

    theta = theta + kappa * (theta_new - theta);

    % compute phi
    % what happens if theta_tmp_mat is 0/0
    phi_tmp_mat_1 = theta_tmp_mat;
    phi_tmp_mat_2 = theta_tmp_mat;
    for (i=1:num_cluster)
        phi_tmp_mat_1(:,i) = phi_tmp_mat_1(:,i) .* (r_hat-theta(i)).^2;
    end
    phi_new = sum(phi_tmp_mat_1) ./ sum(phi_tmp_mat_2) - tau_r ;
    phi_new(zero_idx) = phi(zero_idx);  % keep the gaussian mixtures with zero weight fixed

    phi_new_inf_idx = isinf(phi_new);   % guard against the case of dision by 0
    phi_new(phi_new_inf_idx) = phi(phi_new_inf_idx); % keep the gaussian mixtures with infinity phi fixed just in case

    phi_new = phi_new';
    for (i=1:num_cluster)
        if (phi_new(i)<0)
            phi_new(i) = phi(i);
        end
    end
    
    phi = phi + kappa * (phi_new - phi);

    % compute psi
    psi_tmp_mat = lambda_block_3;
    psi_tmp_mat_1 = psi_tmp_mat;
    psi_tmp_mat_2 = psi_tmp_mat;
    psi_new = sum(psi_tmp_mat_1 .* (r_hat.^2)) / sum(psi_tmp_mat_2) - tau_r;

    if (psi_new<0)
        psi_new = psi;
    end

    psi = psi + kappa * (psi_new - psi);

end


function tau_w = output_parameter_est(y_lower, y_upper, tau_w, p_hat, tau_p, kappa)

    tau_w_pre = tau_w;
    
    % so many complicated steps just to prevent NAN, why doesn't MATLAb have a clever way to compute 0*Inf???
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

    y_upper_bar_minus_p_bar_tri = y_upper_bar_minus_p_bar.^3;
    y_lower_bar_minus_p_bar_tri = y_lower_bar_minus_p_bar.^3;
    mut_tri_exp_y_upper_bar_minus_p_bar = y_upper_bar_minus_p_bar_tri.*exp_y_upper_bar_minus_p_bar;
    mut_tri_exp_y_upper_bar_minus_p_bar((abs(y_upper_bar_minus_p_bar_tri)==Inf)&(exp_y_upper_bar_minus_p_bar==0))=0;
    mut_tri_exp_y_upper_bar_minus_p_bar((y_upper_bar_minus_p_bar_tri==0)&(abs(exp_y_upper_bar_minus_p_bar)==Inf))=0;
    mut_tri_exp_y_lower_bar_minus_p_bar = y_lower_bar_minus_p_bar_tri.*exp_y_lower_bar_minus_p_bar;
    mut_tri_exp_y_lower_bar_minus_p_bar((abs(y_lower_bar_minus_p_bar_tri)==Inf)&(exp_y_lower_bar_minus_p_bar==0))=0;
    mut_tri_exp_y_lower_bar_minus_p_bar((y_lower_bar_minus_p_bar_tri==0)&(abs(exp_y_lower_bar_minus_p_bar)==Inf))=0;

    %exp_diff_block = exp(-0.5*(y_upper_bar_minus_p_bar).^2)-exp(-0.5*(y_lower_bar_minus_p_bar).^2);
    %exp_diff_block_mod = (y_upper_bar_minus_p_bar).*exp(-0.5*(y_upper_bar_minus_p_bar).^2)-(y_lower_bar_minus_p_bar).*exp(-0.5*(y_lower_bar_minus_p_bar).^2);
    %exp_diff_block_mod_2 = ((y_upper_bar_minus_p_bar).^3).*exp(-0.5*(y_upper_bar_minus_p_bar).^2)-((y_lower_bar_minus_p_bar).^3).*exp(-0.5*(y_lower_bar_minus_p_bar).^2);

    %exp_diff_block = exp_y_upper_bar_minus_p_bar - exp_y_lower_bar_minus_p_bar;
    exp_diff_block_mod = mut_exp_y_upper_bar_minus_p_bar - mut_exp_y_lower_bar_minus_p_bar;
    exp_diff_block_mod_2 = mut_tri_exp_y_upper_bar_minus_p_bar - mut_tri_exp_y_lower_bar_minus_p_bar;

    PI_0 = 0.5*(erf(sqrt(0.5)*(y_upper_bar_minus_p_bar))-erf(sqrt(0.5)*(y_lower_bar_minus_p_bar))); % to improve the precision
    PI_0(abs(PI_0)<eps) = eps;

    par_first = -1/(2*sqrt(2*pi))/(tau_w+tau_p)*exp_diff_block_mod;
    par_second = 3/(4*sqrt(2*pi))/((tau_w+tau_p)^2)*exp_diff_block_mod - 1/(4*sqrt(2*pi))/((tau_w+tau_p)^2)*exp_diff_block_mod_2;

    pr_tau_w_first = sum(1./PI_0.*par_first);
    pr_tau_w_second = sum(-1./(PI_0.^2).*(par_first.^2) + 1./PI_0.*par_second);

    if (pr_tau_w_second<0)
        tau_w_new = tau_w - pr_tau_w_first/pr_tau_w_second;
    else
        %tau_w_new = tau_w;
        if (pr_tau_w_first>0)
            tau_w_new = tau_w*1.1;
        else
            tau_w_new = tau_w*0.9;
        end
    end

    tau_w = tau_w + kappa * (tau_w_new - tau_w);

    tau_w = max(tau_w, 1e-12);

end
