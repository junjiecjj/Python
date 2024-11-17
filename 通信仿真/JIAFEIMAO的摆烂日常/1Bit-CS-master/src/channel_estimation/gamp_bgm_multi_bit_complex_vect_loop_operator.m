function [res, input_par, output_par] = gamp_bgm(A, y, tune_idx, gamp_par, input_par, output_par)

    % the operator version, i.e. A is an operator here

    % use MMSE formulation of GAMP with scalar variance
    % the prior distribution of the input channel is Bernoulli-Gaussian mixture
    % the prior distribution of the output channel is white-Gaussian
    M   = size(A,1);    % the dimensionality of the measurement y
    N   = size(A,2);    % the dimensionality of the signal x
    
    
    % set GAMP parameters
    max_pe_inner_ite = gamp_par.max_pe_inner_ite; % maximum number of gamp iterations
    max_pe_ite = gamp_par.max_pe_ite;   % maximum number of parameter estimation iterations
    cvg_thd = gamp_par.cvg_thd; % the convergence threshold
    kappa   = gamp_par.kappa;   % learning rate or damping rate
    eta     = gamp_par.eta;     % damping rate for the gamp iterations
    
    x_hat   = gamp_par.x_hat;   % estimated signal
    tau_x   = gamp_par.tau_x;   % signal variance
    s_hat   = gamp_par.s_hat;   % dummy variable from output function
    
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
    real_quant_thd = output_par.real_quant_thd;
    imag_quant_thd = output_par.imag_quant_thd;
    
    % process y into lower and upper theshold based on quant_thd
    % note that y is supposed to go from 2 to 2^K+1 continuously
    y_real_lower = zeros(length(y),1);
    y_real_upper = zeros(length(y),1);
    y_imag_lower = zeros(length(y),1);
    y_imag_upper = zeros(length(y),1);
    for (i=1:length(y))
        y_real_lower(i) = real_quant_thd(real(y(i))-1);
        y_real_upper(i) = real_quant_thd(real(y(i)));
        y_imag_lower(i) = imag_quant_thd(imag(y(i))-1);
        y_imag_upper(i) = imag_quant_thd(imag(y(i)));
    end

    % set the lower bound and upper bound to -inf and +inf
    % why am i doing this ...
    y_real_lower(1) = -Inf;
    y_real_upper(length(y_real_upper)) = Inf;
    y_imag_lower(1) = -Inf;
    y_imag_upper(length(y_imag_upper)) = Inf;

    tune_vect = de2bi(tune_idx);
    if (length(tune_vect)<8)
        tune_vect = [tune_vect repmat(0, 1, 8-length(tune_vect))];
    end

    tau_p = A.multSq(tau_x);
    p_hat = A.mult(x_hat) - tau_p .* s_hat;

    for (ite_pe = 1:max_pe_ite)

        x_hat_pe_pre = x_hat;
        
        % output nonlinear step
        if (ite_pe>1)
            s_hat_pre = s_hat;
            tau_s_pre = tau_s;
        end
        [s_hat, tau_s] = output_function(p_hat, tau_p, y_real_lower, y_real_upper, y_imag_lower, y_imag_upper, tau_w);
        %fprintf('s: %d\t%d\n', mean(s_hat), mean(tau_s))
        if (ite_pe>1)
            if (tune_vect(4)==1)
                s_hat = s_hat_pre + eta*(s_hat-s_hat_pre);
            end
            if (tune_vect(3)==1)
                tau_s = tau_s_pre + eta*(tau_s-tau_s_pre);
            end
        end

        tau_s(tau_s<eps) = eps;

        % input linear step
        if (ite_pe>1)
            tau_r_pre = tau_r;
            r_hat_pre = r_hat;
        end

        tau_r = 1 ./ A.multSqTr(tau_s);
	    r_hat = x_hat + tau_r .* A.multTr(s_hat);	% this is conjungate transpose
        %fprintf('r: %d\t%d\n', mean(r_hat), mean(tau_r))
        if (ite_pe>1)
            if (tune_vect(5)==1)
                tau_r = tau_r_pre + eta*(tau_r-tau_r_pre);
            end
            if (tune_vect(6)==1)
                r_hat = r_hat_pre + eta*(r_hat-r_hat_pre);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% pre set the outlier distribution variance %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        psi = var(r_hat(abs(r_hat)>eps));   % should we treat real and imaginary parts separately???

        % parameter estimation
        for (ite_par = 1:max_pe_inner_ite)
            [lambda, omega, theta, phi, gamma, psi] = input_parameter_est(r_hat, tau_r, lambda, omega, theta, phi, gamma, psi, kappa); 
        end

        % input nonlinear step
        if (ite_pe>1)
            x_hat_pre = x_hat;
            tau_x_pre = tau_x;
        end

        [x_hat, tau_x] = input_function(r_hat, tau_r, lambda, omega, theta, phi, gamma, psi);
        %fprintf('x: %d\t%d\n', mean(x_hat), mean(tau_x))
        if (ite_pe>1)
            if (tune_vect(8)==1)
                x_hat = x_hat_pre + eta*(x_hat-x_hat_pre);
            end
            if (tune_vect(7)==1)
                tau_x = tau_x_pre + eta*(tau_x-tau_x_pre);
            end
        end


        % output linear step
        if (ite_pe>1)
            tau_p_pre = tau_p;
            p_hat_pre = p_hat;
        end

        tau_p = A.multSq(tau_x);
        p_hat = A.mult(x_hat) - tau_p .* s_hat;
        %fprintf('p: %d\t%d\n', mean(p_hat), mean(tau_p))
        if (ite_pe>1)
            if (tune_vect(1)==1)
                tau_p = tau_p_pre + eta*(tau_p-tau_p_pre);
            end
            if (tune_vect(2)==1)
                p_hat = p_hat_pre + eta*(p_hat-p_hat_pre);
            end
        end


        % parameter estimation
        for (ite_par = 1:max_pe_inner_ite)
            tau_w = output_parameter_est(y_real_lower, y_real_upper, y_imag_lower, y_imag_upper, tau_w, p_hat, tau_p, kappa);
            %tau_w
        end
        %tau_w

        cvg_val_pe = norm(x_hat-x_hat_pe_pre) / norm(x_hat);
        
        fprintf('Iteration %d: %d\n', ite_pe, cvg_val_pe);

        if (cvg_val_pe<cvg_thd) 
            fprintf('Convergence reached\n');
            fprintf('Lambda: %f\n', lambda);
            fprintf('Omega: ');
            for (i=1:num_c)
                fprintf('%f ', omega(i))
            end
            fprintf('\n')

            fprintf('Theta: ');
            for (i=1:num_c)
                fprintf('%f + %f i ', real(theta(i)), imag(theta(i)))
            end
            fprintf('\n')

            fprintf('Phi: ')
            for (i=1:num_c)
                fprintf('%f ', phi(i))
            end
            fprintf('\n')

            fprintf('Tau: %f\n', tau_w);
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
    input_par.gamma = gamma;
    input_par.psi   = psi;
    
    % update output distribution parameters
    output_par.tau_w = tau_w; % the white-Gaussian noise variance

end

function [x_hat, tau_x] = input_function(r_hat, tau_r, lambda, omega, theta, phi, gamma, psi)
    
    dim_smp=length(r_hat);
    num_cluster=length(omega);

    block_mat = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        block_mat(:,i) = (1-gamma) * omega(i) * (tau_r./(phi(i)+tau_r)) .* exp( -(abs(theta(i)-r_hat) ./ sqrt(phi(i)+tau_r)).^2 );
    end

    % compute x_hat
    block_mat_nmr_x = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x(:,i) = block_mat_nmr_x(:,i) .* (theta(i) * tau_r + r_hat * phi(i)) ./ (phi(i) + tau_r);
    end

    block_mat_2 = gamma * (tau_r./(psi+tau_r)) .* exp( -(abs(r_hat) ./ sqrt(psi+tau_r)).^2 );
    block_mat_nrm_x_2 = block_mat_2 .* (r_hat*psi ./ (psi+tau_r));

    nmr_x = sum(block_mat_nmr_x, 2) + block_mat_nrm_x_2;
    dnm_x = sum(block_mat, 2) + block_mat_2 + ((1-lambda)/lambda) * exp( -(abs(r_hat) ./ sqrt(tau_r)).^2 );

    x_hat = nmr_x ./ (dnm_x);
    
    % if dnm_x is zero, set x_hat to r_hat
    dnm_x_zero_idx = (dnm_x==0);
    x_hat(dnm_x_zero_idx) = r_hat(dnm_x_zero_idx);

    % compute tau_x
    block_mat_nmr_x_sq = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x_sq(:,i) = block_mat_nmr_x_sq(:,i) .* (  phi(i)*tau_r./(phi(i) + tau_r) + (abs( (theta(i)*tau_r + r_hat*phi(i)) ./ (phi(i) + tau_r) )).^2  );
    end

    block_mat_nmr_x_sq_2 = block_mat_2 .* ( psi*tau_r./(psi+tau_r) + (abs(r_hat*psi) ./ (psi+tau_r)).^2 );

    nmr_x_sq = sum(block_mat_nmr_x_sq, 2) + block_mat_nmr_x_sq_2;
    dnm_x_sq = dnm_x;

    tau_x_seq = (nmr_x_sq ./ dnm_x_sq - (abs(x_hat)).^2);  % this is nonnegative in theory

    % if dnm_x is zero, set tau_x_seq to 0
    tau_x_seq(dnm_x_zero_idx) = 0;
    
    tau_x = repmat(mean(tau_x_seq),size(tau_x_seq));
    tau_x = max(tau_x, 1e-12);  % just in case

end

function [s_hat, tau_s] = output_function(p_hat, tau_p, y_real_lower, y_real_upper, y_imag_lower, y_imag_upper, tau_w)

    % divide tau_w and tau_p by 2
    tau_w = 0.5*tau_w;
    tau_p = 0.5*tau_p;

    p_hat_real = real(p_hat);
    p_hat_imag = imag(p_hat);
    % so many complicated steps just to prevent NAN, why doesn't MATLAb have a clever way to compute 0*Inf???	
    p_bar = p_hat./sqrt(tau_w+tau_p);    % should this be divided by 2
    p_real_bar = real(p_bar);
    p_imag_bar = imag(p_bar);


    y_real_lower_bar = y_real_lower./sqrt(tau_w+tau_p);
    y_real_upper_bar = y_real_upper./sqrt(tau_w+tau_p);
    y_real_upper_bar_minus_p_bar = y_real_upper_bar-p_real_bar;
    y_real_lower_bar_minus_p_bar = y_real_lower_bar-p_real_bar;
    exp_y_real_upper_bar_minus_p_bar = exp(-0.5*(y_real_upper_bar_minus_p_bar).^2);
    exp_y_real_lower_bar_minus_p_bar = exp(-0.5*(y_real_lower_bar_minus_p_bar).^2);
    mut_exp_y_real_upper_bar_minus_p_bar = (y_real_upper_bar_minus_p_bar).*exp_y_real_upper_bar_minus_p_bar;
    mut_exp_y_real_upper_bar_minus_p_bar((abs(y_real_upper_bar_minus_p_bar)==Inf)&(exp_y_real_upper_bar_minus_p_bar==0))=0;
    mut_exp_y_real_upper_bar_minus_p_bar((y_real_upper_bar_minus_p_bar==0)&(abs(exp_y_real_upper_bar_minus_p_bar)==Inf))=0;
    mut_exp_y_real_lower_bar_minus_p_bar = (y_real_lower_bar_minus_p_bar).*exp_y_real_lower_bar_minus_p_bar;
    mut_exp_y_real_lower_bar_minus_p_bar((abs(y_real_lower_bar_minus_p_bar)==Inf)&(exp_y_real_lower_bar_minus_p_bar==0))=0;
    mut_exp_y_real_lower_bar_minus_p_bar((y_real_lower_bar_minus_p_bar==0)&(abs(exp_y_real_lower_bar_minus_p_bar)==Inf))=0;


    exp_diff_real_block = exp_y_real_upper_bar_minus_p_bar - exp_y_real_lower_bar_minus_p_bar;
    exp_diff_real_block_mod = mut_exp_y_real_upper_bar_minus_p_bar - mut_exp_y_real_lower_bar_minus_p_bar;

    %sum(y_real_upper_bar_minus_p_bar)
    %sum(y_real_lower_bar_minus_p_bar)

    PI_0_real = 0.5*(erf(sqrt(0.5)*(y_real_upper_bar_minus_p_bar))-erf(sqrt(0.5)*(y_real_lower_bar_minus_p_bar))); % to improve the precision
    PI_0_real(abs(PI_0_real)<eps) = eps;

    PI_1_real = p_hat_real.*PI_0_real - tau_p./sqrt(2*pi*(tau_w+tau_p)).*exp_diff_real_block;

    PI_2_real = (p_hat_real.^2+tau_p).*PI_0_real - (1/sqrt(2*pi))./(tau_w+tau_p).*(tau_p.^2).*exp_diff_real_block_mod - 2/sqrt(2*pi)*tau_p.*p_real_bar.*exp_diff_real_block;

    z_hat_real = PI_1_real./PI_0_real;
    z_hat_sq_real = PI_2_real./PI_0_real;


    y_imag_lower_bar = y_imag_lower./sqrt(tau_w+tau_p);
    y_imag_upper_bar = y_imag_upper./sqrt(tau_w+tau_p);
    y_imag_upper_bar_minus_p_bar = y_imag_upper_bar-p_imag_bar;
    y_imag_lower_bar_minus_p_bar = y_imag_lower_bar-p_imag_bar;
    exp_y_imag_upper_bar_minus_p_bar = exp(-0.5*(y_imag_upper_bar_minus_p_bar).^2);
    exp_y_imag_lower_bar_minus_p_bar = exp(-0.5*(y_imag_lower_bar_minus_p_bar).^2);
    mut_exp_y_imag_upper_bar_minus_p_bar = (y_imag_upper_bar_minus_p_bar).*exp_y_imag_upper_bar_minus_p_bar;
    mut_exp_y_imag_upper_bar_minus_p_bar((abs(y_imag_upper_bar_minus_p_bar)==Inf)&(exp_y_imag_upper_bar_minus_p_bar==0))=0;
    mut_exp_y_imag_upper_bar_minus_p_bar((y_imag_upper_bar_minus_p_bar==0)&(abs(exp_y_imag_upper_bar_minus_p_bar)==Inf))=0;
    mut_exp_y_imag_lower_bar_minus_p_bar = (y_imag_lower_bar_minus_p_bar).*exp_y_imag_lower_bar_minus_p_bar;
    mut_exp_y_imag_lower_bar_minus_p_bar((abs(y_imag_lower_bar_minus_p_bar)==Inf)&(exp_y_imag_lower_bar_minus_p_bar==0))=0;
    mut_exp_y_imag_lower_bar_minus_p_bar((y_imag_lower_bar_minus_p_bar==0)&(abs(exp_y_imag_lower_bar_minus_p_bar)==Inf))=0;


    exp_diff_imag_block = exp_y_imag_upper_bar_minus_p_bar - exp_y_imag_lower_bar_minus_p_bar;
    exp_diff_imag_block_mod = mut_exp_y_imag_upper_bar_minus_p_bar - mut_exp_y_imag_lower_bar_minus_p_bar;

    PI_0_imag = 0.5*(erf(sqrt(0.5)*(y_imag_upper_bar_minus_p_bar))-erf(sqrt(0.5)*(y_imag_lower_bar_minus_p_bar))); % to improve the precision
    PI_0_imag(abs(PI_0_imag)<eps) = eps;

    PI_1_imag = p_hat_imag.*PI_0_imag - tau_p./sqrt(2*pi*(tau_w+tau_p)).*exp_diff_imag_block;

    PI_2_imag = (p_hat_imag.^2+tau_p).*PI_0_imag - (1/sqrt(2*pi))./(tau_w+tau_p).*(tau_p.^2).*exp_diff_imag_block_mod - 2/sqrt(2*pi)*tau_p.*p_imag_bar.*exp_diff_imag_block;

    z_hat_imag = PI_1_imag./PI_0_imag;
    z_hat_sq_imag = PI_2_imag./PI_0_imag;

    z_hat = complex(z_hat_real, z_hat_imag);
    z_hat_sq = z_hat_sq_real + z_hat_sq_imag;


    tau_z = z_hat_sq-abs(z_hat).^2;

    % multiply tau_w and tau_p by 2
    tau_w = 2*tau_w;
    tau_p = 2*tau_p;

    s_hat = 1./tau_p.*(z_hat-p_hat);
    tau_s = 1./tau_p.*(1-tau_z./tau_p);

    tau_s = repmat(mean(tau_s),size(tau_s));

end

function [lambda, omega, theta, phi, gamma, psi] = input_parameter_est(r_hat, tau_r, lambda, omega, theta, phi, gamma, psi, kappa)
   
    lambda_pre = lambda;
    omega_pre = omega;
    theta_pre = theta;
    phi_pre = phi;
    gamma_pre = gamma;
    psi_pre = psi;

    % do we need some kind of normalization here?
    dim_smp=length(r_hat);
    num_cluster=length(omega);

    lambda_tmp_mat_1 = zeros(dim_smp, num_cluster);
    for (i = 1:num_cluster)
        lambda_tmp_mat_1(:,i) = lambda * (1-gamma) * omega(i) * 1./(tau_r+phi(i)) .* exp( - (abs(r_hat-theta(i)) ./ sqrt(tau_r+phi(i))).^2 );
    end

    lambda_tmp_3 = lambda * gamma * 1./(tau_r+psi) .* exp( -(abs(r_hat) ./ sqrt(tau_r+psi)).^2);

    % compute lambda
    lambda_tmp_1 = sum(lambda_tmp_mat_1, 2);
    lambda_tmp_2 = (1-lambda) * 1 ./ tau_r .* exp( - (abs(r_hat) ./ sqrt(tau_r)).^2);

    lambda_tmp_sum = lambda_tmp_1 + lambda_tmp_2 + lambda_tmp_3;
    lambda_tmp_sum_zero_idx = (lambda_tmp_sum==0);    % leave the outliers to the first Gaussian component


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
    omega_new = omega_new.';	% this is transpose not conjungate transpose
    omega_new = omega_new / sum(omega_new);
    
    omega = omega + kappa * (omega_new - omega);

    % compute gamma
    gamma_new = sum(lambda_block_3) / (sum(sum(lambda_block_1)) + sum(lambda_block_3));

    %gamma_new
    gamma = gamma + kappa * (gamma_new - gamma);

    
    % compute theta
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

    theta_new_inf_idx = isinf(theta_new);
    theta_new(theta_new_inf_idx) = theta(theta_new_inf_idx); % keep the gaussian mixtures with infinity theta fixed just in case

    theta_new = theta_new.';	% this is transpose, not conjungate transpose
    
    theta = theta + kappa * (theta_new - theta);
    
    % compute phi
    % what happens if theta_tmp_mat is 0/0
    phi_tmp_mat_1 = theta_tmp_mat;
    phi_tmp_mat_2 = theta_tmp_mat;
    for (i=1:num_cluster)
        phi_tmp_mat_1(:,i) = phi_tmp_mat_1(:,i) .* (abs(r_hat-theta(i))).^2;
    end
    phi_new = sum(phi_tmp_mat_1) ./ sum(phi_tmp_mat_2) - tau_r(1);   % since the coefficients in tau_r are the same, just use tau_r(1)
    phi_new(zero_idx) = phi(zero_idx);  % keep the gaussian mixtures with zero weight fixed

    phi_new_inf_idx = isinf(phi_new);
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
    psi_new = sum(psi_tmp_mat_1 .* (abs(r_hat).^2)) / sum(psi_tmp_mat_2) - tau_r(1);    % since the coefficients in tau_r are the same, just use tau_r(1)

    if (psi_new<0)
        psi_new = psi;
    end
    
    psi = psi + kappa * (psi_new - psi);

end

function tau_w = output_parameter_est(y_real_lower, y_real_upper, y_imag_lower, y_imag_upper, tau_w, p_hat, tau_p, kappa)

    tau_w_pre = tau_w;

    % divide the tau_p and tau_w by 2 here
    tau_w = 0.5*tau_w;
    tau_p = 0.5*tau_p;

    p_hat_real = real(p_hat);
    p_hat_imag = imag(p_hat);
    % so many complicated steps just to prevent NAN, why doesn't MATLAb have a clever way to compute 0*Inf???
    p_bar = p_hat./sqrt(tau_w+tau_p);
    p_real_bar = real(p_bar);
    p_imag_bar = imag(p_bar);
    
    y_real_lower_bar = y_real_lower./sqrt(tau_w+tau_p);
    y_real_upper_bar = y_real_upper./sqrt(tau_w+tau_p);
    y_real_upper_bar_minus_p_bar = y_real_upper_bar-p_real_bar;
    y_real_lower_bar_minus_p_bar = y_real_lower_bar-p_real_bar;
    exp_y_real_upper_bar_minus_p_bar = exp(-0.5*(y_real_upper_bar_minus_p_bar).^2);
    exp_y_real_lower_bar_minus_p_bar = exp(-0.5*(y_real_lower_bar_minus_p_bar).^2);
    mut_exp_y_real_upper_bar_minus_p_bar = (y_real_upper_bar_minus_p_bar).*exp_y_real_upper_bar_minus_p_bar;
    mut_exp_y_real_upper_bar_minus_p_bar((abs(y_real_upper_bar_minus_p_bar)==Inf)&(exp_y_real_upper_bar_minus_p_bar==0))=0;
    mut_exp_y_real_upper_bar_minus_p_bar((y_real_upper_bar_minus_p_bar==0)&(abs(exp_y_real_upper_bar_minus_p_bar)==Inf))=0;
    mut_exp_y_real_lower_bar_minus_p_bar = (y_real_lower_bar_minus_p_bar).*exp_y_real_lower_bar_minus_p_bar;
    mut_exp_y_real_lower_bar_minus_p_bar((abs(y_real_lower_bar_minus_p_bar)==Inf)&(exp_y_real_lower_bar_minus_p_bar==0))=0;
    mut_exp_y_real_lower_bar_minus_p_bar((y_real_lower_bar_minus_p_bar==0)&(abs(exp_y_real_lower_bar_minus_p_bar)==Inf))=0;

    y_real_upper_bar_minus_p_bar_tri = y_real_upper_bar_minus_p_bar.^3;
    y_real_lower_bar_minus_p_bar_tri = y_real_lower_bar_minus_p_bar.^3;
    mut_tri_exp_y_real_upper_bar_minus_p_bar = y_real_upper_bar_minus_p_bar_tri.*exp_y_real_upper_bar_minus_p_bar;
    mut_tri_exp_y_real_upper_bar_minus_p_bar((abs(y_real_upper_bar_minus_p_bar_tri)==Inf)&(exp_y_real_upper_bar_minus_p_bar==0))=0;
    mut_tri_exp_y_real_upper_bar_minus_p_bar((y_real_upper_bar_minus_p_bar_tri==0)&(abs(exp_y_real_upper_bar_minus_p_bar)==Inf))=0;
    mut_tri_exp_y_real_lower_bar_minus_p_bar = y_real_lower_bar_minus_p_bar_tri.*exp_y_real_lower_bar_minus_p_bar;
    mut_tri_exp_y_real_lower_bar_minus_p_bar((abs(y_real_lower_bar_minus_p_bar_tri)==Inf)&(exp_y_real_lower_bar_minus_p_bar==0))=0;
    mut_tri_exp_y_real_lower_bar_minus_p_bar((y_real_lower_bar_minus_p_bar_tri==0)&(abs(exp_y_real_lower_bar_minus_p_bar)==Inf))=0;

    exp_diff_real_block_mod = mut_exp_y_real_upper_bar_minus_p_bar - mut_exp_y_real_lower_bar_minus_p_bar;
    exp_diff_real_block_mod_2 = mut_tri_exp_y_real_upper_bar_minus_p_bar - mut_tri_exp_y_real_lower_bar_minus_p_bar;

    PI_0_real = 0.5*(erf(sqrt(0.5)*(y_real_upper_bar_minus_p_bar))-erf(sqrt(0.5)*(y_real_lower_bar_minus_p_bar))); % to improve the precision
    PI_0_real(abs(PI_0_real)<eps) = eps;

    par_first_real = -1/(2*sqrt(2*pi))./(tau_w+tau_p).*exp_diff_real_block_mod;
    par_second_real = 3/(4*sqrt(2*pi))./((tau_w+tau_p).^2).*exp_diff_real_block_mod - 1/(4*sqrt(2*pi))./((tau_w+tau_p).^2).*exp_diff_real_block_mod_2;
    


    y_imag_lower_bar = y_imag_lower./sqrt(tau_w+tau_p);
    y_imag_upper_bar = y_imag_upper./sqrt(tau_w+tau_p);
    y_imag_upper_bar_minus_p_bar = y_imag_upper_bar-p_imag_bar;
    y_imag_lower_bar_minus_p_bar = y_imag_lower_bar-p_imag_bar;
    exp_y_imag_upper_bar_minus_p_bar = exp(-0.5*(y_imag_upper_bar_minus_p_bar).^2);
    exp_y_imag_lower_bar_minus_p_bar = exp(-0.5*(y_imag_lower_bar_minus_p_bar).^2);
    mut_exp_y_imag_upper_bar_minus_p_bar = (y_imag_upper_bar_minus_p_bar).*exp_y_imag_upper_bar_minus_p_bar;
    mut_exp_y_imag_upper_bar_minus_p_bar((abs(y_imag_upper_bar_minus_p_bar)==Inf)&(exp_y_imag_upper_bar_minus_p_bar==0))=0;
    mut_exp_y_imag_upper_bar_minus_p_bar((y_imag_upper_bar_minus_p_bar==0)&(abs(exp_y_imag_upper_bar_minus_p_bar)==Inf))=0;
    mut_exp_y_imag_lower_bar_minus_p_bar = (y_imag_lower_bar_minus_p_bar).*exp_y_imag_lower_bar_minus_p_bar;
    mut_exp_y_imag_lower_bar_minus_p_bar((abs(y_imag_lower_bar_minus_p_bar)==Inf)&(exp_y_imag_lower_bar_minus_p_bar==0))=0;
    mut_exp_y_imag_lower_bar_minus_p_bar((y_imag_lower_bar_minus_p_bar==0)&(abs(exp_y_imag_lower_bar_minus_p_bar)==Inf))=0;

    y_imag_upper_bar_minus_p_bar_tri = y_imag_upper_bar_minus_p_bar.^3;
    y_imag_lower_bar_minus_p_bar_tri = y_imag_lower_bar_minus_p_bar.^3;
    mut_tri_exp_y_imag_upper_bar_minus_p_bar = y_imag_upper_bar_minus_p_bar_tri.*exp_y_imag_upper_bar_minus_p_bar;
    mut_tri_exp_y_imag_upper_bar_minus_p_bar((abs(y_imag_upper_bar_minus_p_bar_tri)==Inf)&(exp_y_imag_upper_bar_minus_p_bar==0))=0;
    mut_tri_exp_y_imag_upper_bar_minus_p_bar((y_imag_upper_bar_minus_p_bar_tri==0)&(abs(exp_y_imag_upper_bar_minus_p_bar)==Inf))=0;
    mut_tri_exp_y_imag_lower_bar_minus_p_bar = y_imag_lower_bar_minus_p_bar_tri.*exp_y_imag_lower_bar_minus_p_bar;
    mut_tri_exp_y_imag_lower_bar_minus_p_bar((abs(y_imag_lower_bar_minus_p_bar_tri)==Inf)&(exp_y_imag_lower_bar_minus_p_bar==0))=0;
    mut_tri_exp_y_imag_lower_bar_minus_p_bar((y_imag_lower_bar_minus_p_bar_tri==0)&(abs(exp_y_imag_lower_bar_minus_p_bar)==Inf))=0;

    exp_diff_imag_block_mod = mut_exp_y_imag_upper_bar_minus_p_bar - mut_exp_y_imag_lower_bar_minus_p_bar;
    exp_diff_imag_block_mod_2 = mut_tri_exp_y_imag_upper_bar_minus_p_bar - mut_tri_exp_y_imag_lower_bar_minus_p_bar;

    PI_0_imag = 0.5*(erf(sqrt(0.5)*(y_imag_upper_bar_minus_p_bar))-erf(sqrt(0.5)*(y_imag_lower_bar_minus_p_bar))); % to improve the precision
    PI_0_imag(abs(PI_0_imag)<eps) = eps;

    par_first_imag = -1/(2*sqrt(2*pi))./(tau_w+tau_p).*exp_diff_imag_block_mod;
    par_second_imag = 3/(4*sqrt(2*pi))./((tau_w+tau_p).^2).*exp_diff_imag_block_mod - 1/(4*sqrt(2*pi))./((tau_w+tau_p).^2).*exp_diff_imag_block_mod_2;
    
        
    pr_tau_w_first = sum(1./PI_0_real.*par_first_real) + sum(1./PI_0_imag.*par_first_imag);
    pr_tau_w_second = sum(-1./(PI_0_real.^2).*(par_first_real.^2) + 1./PI_0_real.*par_second_real) + sum(-1./(PI_0_imag.^2).*(par_first_imag.^2) + 1./PI_0_imag.*par_second_imag);

    if (pr_tau_w_second<0)
        tau_w_new = tau_w - pr_tau_w_first/pr_tau_w_second;
    else
        if (pr_tau_w_first>0)
            tau_w_new = tau_w*1.1;
        else
            tau_w_new = tau_w*0.9;
        end
    end

    % multiply tau_w_new tau_w tau_p by 2
    tau_w_new = 2*tau_w_new;
    tau_w = 2*tau_w;
    tau_p = 2*tau_p;

    tau_w = tau_w + kappa * (tau_w_new - tau_w);

    tau_w = max(tau_w, 1e-12);
end
