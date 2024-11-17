snr_val = 30;	% signal to noise level
Np_val = 4096;	%  Pilot block length
bit_num = 1;	% the number of bits
LN_num = 2;		% number of computing threads
LN = maxNumCompThreads(LN_num);    
tune_idx = 128;	% the damping index for signal recovery: damping operation on x_hat
verbose = 0;	% 1: print convergence info at every iteration; 0: do not print convergence info at every iteration

% PE-AMP parameters
num_c = 3;	% number of Gaussian mixture components
max_pe_inner_ite = 20;	% maximum number of inner iterations to estimate the parameters 
max_pe_ite = 20;	% the maximum number of iterations for AMP-PE and AMP-AWGN
cvg_thd = 1e-6; % convergence threshold
kappa = 0.1;      % [0,1]: damping rate of parameter estimation
eta = 0.5;		% [0,1]: damping rate of AMP iterations


% Broadband mmWave Channel Model: simulation setup
Nta = 8;  % azimuth
Nte = 8;  % elevation
Nt = Nta * Nte;
Nra = 8;  % azimuth
Nre = 8;  % elevation
Nr = Nra * Nre;
Nd = 16;  % Delay spread L
debias = 1; %true;  % only for 1 bit CS where the norm info is lost


% parameters used to generate channel coeffient
Nt_W = Nta;
Nt_H = Nte;
Nr_W = Nra;
Nr_H = Nre;
Angle_spread_W = pi/24;
Angle_spread_H = pi/24;
Num_cluster = 4;	% the number of clusters
Num_ray = 10;		% the number of paths
Narrowband_flag = 0;    % Broadband


%% Basis matrices
Br = 1/sqrt(Nr) * kron( dftmtx(Nra), dftmtx(Nre) );
Bt = 1/sqrt(Nt) * kron( dftmtx(Nta), dftmtx(Nte) );

%% ADC setup
bit_vector = bit_num; %[1 2 3 4 +inf]; % [1, 2, 3, 4, 5, 6, 7, 8, +inf]; %[1 2 3 4 5 6 7 8 +inf];

% distortion facotor for the Additive Quantization Noise Model
beta_vector = [0.3634 0.1188 0.03744 0.01154 0.003504 0.001035 0.0002999 0.00008543];

%% quantization stepsize
%Lloyd_stepsize_vector = [1.5956 0.9957 0.586 0.3352 0.1881 0.1041 0.0569 0.0308];
%stepsize_scale_factor = 2;


%% Get an estimate of the maximum y value: max_y_val, then compute quantization step size
max_y_val = 30;
stepsize = max_y_val/(2^(bit_num-1));


%% Pilot block length
Np_vector = Np_val; %512*(2:1:10); %[251 509 1021 1531 2039 2557 3067 3583 4093];

%% Training length
%Nc = 512*20;

%% SNR
SNRdB_vector = snr_val; %linspace(-10, 40, 6); %linspace(-10, 40, 11);

%% number of channels for averaging
Num_chan = 1;

%% number of subcarrier
N_sc = 16;

%% Choice of training sequences
% Sequences = 'Random_Gaussian';
 Sequences = 'Random_QPSK';
% Sequences = 'Shifted_Golay';
% Sequences = 'Shifted_ZC';

% Sequences = 'Shifted_Pulse';
% Sequences = 'Shifted_QPSK';
% Sequences = 'Kronecker';
% Sequences = 'Golay';

Params.Sequences = Sequences;

Params.Nd = Nd; 
Params.Nta = Nta;
Params.Nte = Nte;
Params.Nt = Nt; 
Params.Nra = Nra;
Params.Nre = Nre;
Params.Nr = Nr; 
%Params.Nc = Nc; 
Params.N_sc = N_sc;
Params.Br = Br; 
Params.Bt = Bt; 

Params.bit_vector = bit_vector;
Params.SNRdB_vector = SNRdB_vector;
Params.Np_vector = Np_vector;
Params.Num_chan = Num_chan;

if ( strcmp(Params.Sequences, 'Shifted_ZC')||...
        strcmp(Params.Sequences, 'Shifted_QPSK')||strcmp(Params.Sequences,'Shifted_Pulse')...
        || strcmp(Params.Sequences, 'Shifted_Golay'))
    Params.Enable_fast_implementation = 1; % Enable fast implementaion of Ax and A*z;
else
    Params.Enable_fast_implementation = 0; % Disable fast implementaion of Ax and A*z;
end

% save the reconstruction results
nmse_seq_mat = [];
for (trial_num = 1:10)
	nmse_seq = []; 
	fprintf('Trial %d\n', trial_num)

    rng(trial_num)

    % the norm of the channel coefficients
    Gnorm = nan(1, Num_chan);

    for index_channel = 1:1:Num_chan
        % virtual channel model or angular domain channel model
        Hv_all = func_mmWave_Channel_UPA(Nt_W, Nt_H, Nr_W, Nr_H, ...
                     Angle_spread_W, Angle_spread_H, Num_cluster, Num_ray, Narrowband_flag, Nd);
        Gtrue = Hv_all(:,:,:,index_channel);
        gtrue = reshape(Gtrue, [],1);
        
        Gnorm(index_channel) = norm(gtrue);
        
        for index_Np = 1:1:length(Np_vector)
            
            Np = Np_vector(index_Np);
            Params.Np = Np;
            
            % Training sequences in the spatial and time domain, size: Nt X Np
            % Walsh_Hadamard_matrix = hadamard(Np);
            % T_train = Walsh_Hadamard_matrix(randsample(1:Np,Nt), 1:Np);
            
            switch Params.Sequences
                case 'Random_QPSK'                 % random QPSK sequence
                    T_train_unit = sign(randn(Nt,Np))/sqrt(2)+1i*sign(randn(Nt,Np))/sqrt(2);
                    
                case 'Random_Gaussian'
                    T_train_unit = randn(Nt,Np)/sqrt(2) + 1i*randn(Nt,Np)/sqrt(2);
                    
                case 'Random_ZC'
                    T_train_unit = func_ZadoffChuSeq(Nt, Np);
                    
                case 'Golay'
                    if floor(log2(Np)) == log2(Np)
                        [a, b] = generate_golay(log2(Np)-1);
                    else
                        keyboard;
                    end
                    s = [a, b];
                    T0 = zeros(Nt,Np);
                    for i=0:Nt-1
                        %T0(i+1,:) = circshift(s,[0 i*Nd]);
                        T0(i+1,:) = [a(Np/2-i*Nd/2+1: Np/2), a(1:Np/2-i*Nd/2), ...
                            b(Np/2-i*Nd/2+1: Np/2), b(1:Np/2-i*Nd/2)];
                        %for this kind of T0, the singular values are same
                    end
                    T_train_unit = T0;
                    
                case 'Shifted_ZC'
                    % This training sequence design is proposed by Prof. Schniter
                    if mod(Np,2) == 0
                        s = exp((1i*pi/Np)*(0:Np-1).^2);  % even length
                    else
                        s = exp((1i*pi/Np)*(0:Np-1).*((0:Np-1)+1)); % odd length
                    end
                    T0 = zeros(Nt,Np);
                    for i=0:Nt-1
                        T0(i+1,:) = circshift(s,[0 i*Nd]);
                    end
                    T_train_unit = T0;
                    
                case 'Shifted_QPSK'
                    s = sign(randn(1,Np))/sqrt(2)+1i*sign(randn(1,Np))/sqrt(2);
                    T0 = zeros(Nt,Np);
                    for i=0:Nt-1
                        T0(i+1,:) = circshift(s,[0 i*Nd]);
                    end
                    T_train_unit = T0;
                    
                case 'Kronecker'
                    % This training sequence design is proposed by Prof. Schniter
                    s = zeros(1, Np);
                    s(1,1) = 1;
                    T0 = zeros(Nt,Np);
                    for i=0:Nt-1
                        T0(i+1,:) = circshift(s,[0 i*Nd]);
                    end
                    T_train_unit = T0;
                    
                case  'Shifted_Pulse'
                    s = zeros(Nd, Np/Nd);
                    if mod(Np/Nd,2) == 0
                        s(1,:) = exp((1i*pi/ (Np/Nd) )*(0:Np/Nd-1).^2);  % even length
                    else
                        s(1,:) = exp((1i*pi/ (Np/Nd) )*(0:Np/Nd-1).*((0:Np/Nd-1)+1)); % odd length
                    end
                    s = reshape(s, 1, Np);
                    T0 = zeros(Nt,Np);
                    for i=0:Nt-1
                        T0(i+1,:) = circshift(s,[0 i*Nd]);
                    end
                    T_train_unit = T0;
                    
                case  'Shifted_Golay'
                    display('Not well defined!');
                    keyboard;
            end
           

            %% fast construction of T_cyclic by matlab function 'circshift'
            T_cyclic_unit = NaN(Nr*Nd, Np);
            for ii=0:1:Nd-1
                T_cyclic_unit( ii*Nt+1: (ii+1)*Nt, :) = circshift(T_train_unit,ii,2);
            end
            
            % % 1st fast implementation of V_cyclic = kron(eye(Nd), Bt') * T_cyclic
            %T_temp = reshape(T_cyclic, Nt, Nd, Np);
            %V_temp = Bhfast(T_temp, Nt);
            %V_cyclic_1 = reshape(V_temp, Nd*Nt, Np);
            
            % % 2nd fast implementation of V_cyclic = kron(eye(Nd),
            % Bt') * T_cyclic. The time consumptions of 1st and 2nd
            % implementation are similar.
            V_cyclic_unit = NaN(Nr*Nd, Np);
            V_train_unit = Bhfast(T_train_unit, Nt);
            for ii=0:1:Nd-1
                V_cyclic_unit( ii*Nt+1: (ii+1)*Nt, :) = circshift(V_train_unit,ii,2);
            end
            
            for index_bit = 1:1:length(bit_vector)
                
                bit = bit_vector(index_bit);
                if bit~=+inf
                    beta = beta_vector(bit);
                else
                    beta = 0;
                end
                
                %fprintf('Channel Number: %d\n', index_channel)
                %fprintf('Pilot block length: %d\n', Np)
                %fprintf('Number of bits: %d\n', bit)

                for index_SNR = 1:1:length(SNRdB_vector)
                    
                    SNRdB = SNRdB_vector(index_SNR);
                    
                    SNR_linear = 10^(SNRdB/10);
                    
                    % SNR_coeff should be applied to wvar
                    SNR_coeff = 1/(norm(T_train_unit, 'fro'))* sqrt(Np) * sqrt(SNR_linear);
                    
                    T_cyclic  = T_cyclic_unit;
                    V_cyclic = V_cyclic_unit;

                    Params.V_cyclic = V_cyclic;
                    
                    z = reshape( Br * reshape(Gtrue, Nr, []) * V_cyclic, [], 1);
                    
                    % SH: note that the following is different from Mo's code where the wvar is assumed to be known (which is always 1)
                    % SH: wvar is actually standard deviation, anyway, do not want to cause confusion, so keep it as it is...
                    wvar = 1/SNR_coeff; % the noise variance should be changed, because it is unknown in practice
                    
                    y = z + sqrt(1/2)*wvar*randn(Nr*Np,2)*[1;1i];

                    
                    %% norm esimation (seems to be the MMSE estimator??)
                    % compute the variance gain due to the measurement matrix 'A'
                    % gain = trace(T_cyclic * T_cyclic')/Np;
                    
                    gain = SNR_linear * Nd / SNR_coeff^2;
                    
                    % estimate the variance of each element of gtrue
                    % SH: wvar is unknown, which means you can not estimate the norm, right? anyway, i am not using it, so whatever...
                    gvar = ( mean(abs(y).^2) - wvar )/gain;  % is close to SNR_linear/gain
                    
                    % estimate the norm of gtrue (note the dimension of gtrue is Nt*Nr*Nd)
                    estimated_norm = sqrt(gvar * Nt * Nr * Nd);
                    gtrue_norm = norm(gtrue);
                    
                    %% quantization step
                    
                    %% dithering
                    dither_mean = rms([real(y); imag(y)])*0.0; % default value of dithering threshold
                    
                    %% ADC resolution
                    if bit == +inf
                        r_bit = y;
                        stepsize = 0;
                        beta = 0;
                    else
                        %stepsize = stepsize_scale_factor * rms([real(y); imag(y)])* Lloyd_stepsize_vector(bit);
                        
                        r_bit_real = min ( max( floor( (real(y)-dither_mean)/stepsize) + 2^(bit - 1), 0), 2^bit-1) ; %[0 2^bit-1]
                        r_bit_imag = min ( max( floor( (imag(y)-dither_mean)/stepsize )+ 2^(bit - 1), 0), 2^bit-1) ; %[0 2^bit-1]
                        r_bit = r_bit_real + 1j * r_bit_imag;
                        
                    end

                    real_quant_thd = ( -(2^(bit-1)):(2^(bit-1)) )*stepsize + dither_mean;
                    imag_quant_thd = ( -(2^(bit-1)):(2^(bit-1)) )*stepsize + dither_mean;

                    r_bit_real_quant = (r_bit_real - 2^(bit-1) + 0.5)*stepsize + dither_mean;
                    r_bit_imag_quant = (r_bit_imag - 2^(bit-1) + 0.5)*stepsize + dither_mean;

                    % SH: only do this for pe-amp, the quantized symbol is different from before, it starts from 2
                    r_bit_ori = r_bit;	% this is for QIHT only
                    r_bit = (r_bit_real+2) + 1j * (r_bit_imag+2);
                    r_bit_quant = r_bit_real_quant + 1j * r_bit_imag_quant;
                    
                    % define measurement operator
                    % we should be able to make if much faster if we use fft operator instead of fft matrix
                    % why are we redefining V_cyclic here again???
                    %V_train = Bt'*T_cyclic(1:Nt,:);
                    %for ii=0:1:Nd-1
                    %    V_cyclic( ii*Nt+1: (ii+1)*Nt, :) = circshift(V_train,ii,2);
                    %end

                    %% UPA
                    hA = @(in) Afast_Digital_UPA(in, T_cyclic, V_cyclic, Params, 1);
                    hAt = @(in) Afast_Digital_UPA(in, T_cyclic, V_cyclic, Params, 0);

                    %% ULA
                    % hA = @(in) Afast_Digital_ULA(in,S,Nr, 1);
                    % hAt = @(in) Afast_Digital_ULA(in,S,Nr, 0);

                    lA = FxnhandleLinTrans(Nr*Np, Nt*Nr*Nd, hA, hAt); % linear transform object

                    % initialize with l2-min solution via efficient conjungate gradient
                    x_hat = zeros(Nt*Nr*Nd,1);
                    d_cg = zeros(Nt*Nr*Nd,1);
                    d_cg = -hAt(hA(x_hat)-r_bit_quant);
                    r_cg = d_cg;

                    max_cg_ite = 20; 

                    for (ite=1:max_cg_ite)
                        a_cg_n = sum(r_cg'*r_cg);
                        d_cg_tmp = hA(d_cg);
                        a_cg_d = sum(d_cg_tmp'*d_cg_tmp);
                        a_cg = a_cg_n / a_cg_d;
                        x_hat_pre = x_hat;
                        x_hat = x_hat + a_cg*d_cg;
                        cvg_cg_val = norm(x_hat-x_hat_pre,'fro')/norm(x_hat,'fro');
                        if (cvg_cg_val<cvg_thd)
                            break;
                        end

                        r_cg_new = r_cg - a_cg*(hAt(hA(d_cg)));

                        b_cg = sum(r_cg_new'*r_cg_new)/sum(r_cg'*r_cg);
                        d_cg = r_cg_new + b_cg*d_cg;

                        r_cg = r_cg_new;
                    end

                    tau_x = var(x_hat)+1e-12; % initialize scalar signal variance
                    s_hat = zeros(size(r_bit)); % initialize s_hat with all zeros

                    % initialize input distribution parameters
                    lambda = 0.1;

                    % initialize Gaussian mixtures parameters
                    theta=zeros(num_c, 1);  % Gaussian mean
                    phi=ones(num_c, 1);     % Gaussian variance
                    omega=zeros(num_c,1);   % Gaussian mixture weights

                    x_hat_nz = x_hat(abs(x_hat)>=max(abs(x_hat))*0.01);
                    idx_seq = kmeans([real(x_hat_nz) imag(x_hat_nz)], num_c);
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
                    
                    gamma = 0.01;               % the weight of the outlier distribution (a zero-mean Gaussian)
                    psi = var(x_hat_nz)+1e-12;  % the variance of the outlier distribution (a zero-mean Gaussian)

                    % white Gaussian noise variance initialization
                    tau_w = 1e-3;


                    % set GAMP parameters
                    gamp_par.max_pe_inner_ite    = max_pe_inner_ite; % maximum number of gamp iterations
                    gamp_par.max_pe_ite    = max_pe_ite; % maximum number of parameter estimation iterations
                    gamp_par.cvg_thd    = cvg_thd; % the convergence threshold
                    gamp_par.kappa      = kappa;   % learning rate
                    gamp_par.eta        = eta;     % damping rate for gamp

                    gamp_par.x_hat      = zeros(size(x_hat));   % estimated signal
                    gamp_par.tau_x      = repmat(tau_x,size(x_hat));   % signal variance
                    gamp_par.s_hat      = s_hat;   % dummy variable from output function
                    
                    gamp_par.verbose	= verbose;	% verbose option

                    % set input distribution parameters
                    input_par.lambda    = lambda; % Bernoulli parameter
                    input_par.theta     = theta;  % the means of the Gaussian mixtures
                    input_par.phi       = phi;    % the variances of the Gaussian mixtures
                    input_par.omega     = omega;  % the weights of the Gaussian mixtures
                    input_par.num_c     = num_c;  % the number of Gaussian mixtures
                    input_par.gamma     = gamma;  % the weight of the outlier distribution (a zero-mean Gaussian)
                    input_par.psi       = psi;    % the variance of the outlier distribution (a zero-mean Gaussian)

                    % set output distribution parameters
                    output_par.tau_w    = tau_w; % the white-Gaussian noise variance
                    output_par.real_quant_thd = real_quant_thd; % quantization threshold
                    output_par.imag_quant_thd = imag_quant_thd; % quantization threshold

                    %%%%%%%%%%%%%%%%%%%%%
                    %% AMP-PE recovery %%
                    %%%%%%%%%%%%%%%%%%%%%

                    % Run the GAMP algorithm
                    [res, input_par_new, output_par_new] = gamp_bgm_multi_bit_complex_vect_loop_operator(lA, r_bit, tune_idx, gamp_par, input_par, output_par);
                    ghat = res.x_hat;
                    
                    if debias
	                    % only do debias for 1 bit CS since the magnitude information is lost
	                    if (bit==1)
	                        %ghat = estimated_norm * ghat /norm(ghat);
	                        ghat = gtrue_norm * ghat /norm(ghat);
	                    end
                    end
                    
                    nmse_est = -10*log10(norm(gtrue, 'fro')^2/norm(gtrue-ghat, 'fro')^2);
                    nmse_seq = [nmse_seq nmse_est];
                    
                    %%%%%%%%%%%%%%%%%%%%%%%
                    %% AMP-AWGN recovery %%
                    %%%%%%%%%%%%%%%%%%%%%%%
                    [res, input_par_new, output_par_new] = gamp_bgm_complex_vect_loop_operator(lA, r_bit_quant, tune_idx, gamp_par, input_par, output_par);
                    ghat = res.x_hat;
                    
                    if debias
	                    % only do debias for 1 bit CS since the magnitude information is lost
	                    if (bit==1)
	                        %ghat = estimated_norm * ghat /norm(ghat);
	                        ghat = gtrue_norm * ghat /norm(ghat);
	                    end
                    end
                    
                    nmse_est = -10*log10(norm(gtrue, 'fro')^2/norm(gtrue-ghat, 'fro')^2);
                    nmse_seq = [nmse_seq nmse_est];
                    
                    %%%%%%%%%%%%%%%%%%%
                    %% QIHT recovery %%
                    %%%%%%%%%%%%%%%%%%%
                    Par.Nt = Nt;
                    Par.Nta = Nta;
                    Par.Nte = Nte;
                    Par.Nr = Nr;
                    Par.Nra = Nra;
                    Par.Nre = Nre;
                    Par.Nd = Nd;
                    Par.Np = Np;
                    Par.Bt = Bt;
                    Par.Br = Br;
                    Par.Enable_fast_implementation = Params.Enable_fast_implementation;
                    Par.Sequences = Sequences;
                    Par.bit = bit;				% the bit number
                    Par.stepsize = stepsize;	% quantization stepsize
                    Par.beta = beta;
                    Par.SNR_linear = SNR_linear;	% the linear SNR value	
                    Par.dither_mean = dither_mean;
                    Par.alpha = 1e-4;			% QIHT iteration stepsize (needs to be tuned)
                    Par.maxiter = 500;			% maximum number of QIHT iterations
                    Par.stopcrit = 1e-6;		% convergence stopping criterion
                    Par.K2 = round(0.1*length(gtrue));
                    
                    [ ghat ] = func_Broadband_Few_Bit_UPA_QIHT(T_cyclic, r_bit_ori, Par);

                    if debias
	                    % only do debias for 1 bit CS since the magnitude information is lost
	                    if (bit==1)
	                        %ghat = estimated_norm * ghat /norm(ghat);
	                        ghat = gtrue_norm * ghat /norm(ghat);
	                    end
                    end
                    
                    nmse_est = -10*log10(norm(gtrue, 'fro')^2/norm(gtrue-ghat, 'fro')^2);
                    nmse_seq = [nmse_seq nmse_est];
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%
                    %% BP (L1-min) recovery %%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%
                    opA = @(in, mode) Afast_Digital_UPA(in, T_cyclic, V_cyclic, Params, mode);
                    sigma = norm(r_bit_quant - z); 
                    opts = spgSetParms('verbosity',0);
                    
                    [ghat,~,~,info]= spg_bpdn(opA, r_bit_quant, sigma, opts);
                    
                    if debias
	                    % only do debias for 1 bit CS since the magnitude information is lost
	                    if (bit==1)
	                        %ghat = estimated_norm * ghat /norm(ghat);
	                        ghat = gtrue_norm * ghat /norm(ghat);
	                    end
                    end
                    
                    nmse_est = -10*log10(norm(gtrue, 'fro')^2/norm(gtrue-ghat, 'fro')^2);
                    nmse_seq = [nmse_seq nmse_est];

					% save teh nmse values from each random trial to nmse_seq_mat
					nmse_seq_mat = [nmse_seq_mat; nmse_seq];
                    
                    
                end    % end of SNR sequence
            end    % end of bit sequence
        end    % end of Np (the length of training blocks) sequence
    end    % end of the number of channels sequence for averaging
end % end of trial sequence

% plot the nmse values from different approaches
nmse_seq_mean = mean(nmse_seq_mat);
nmse_seq_upper = std(nmse_seq_mat);
nmse_seq_lower = std(nmse_seq_mat);

nmse_x = categorical({'AMP-PE', 'AMP-AWGN', 'QIHT', 'L1-Min'});
nmse_x = reordercats(nmse_x, {'AMP-PE', 'AMP-AWGN', 'QIHT', 'L1-Min'});

figure;
bar(nmse_x, nmse_seq_mean);
ylabel('NMSE (dB)')
title(strcat('Channel Estimation: ', num2str(bit_num), '-bit ADC, Np=', num2str(Np), ', SNR: ', num2str(snr_val), 'dB'))
hold on
er = errorbar(nmse_x, nmse_seq_mean, nmse_seq_lower, nmse_seq_upper);
er.Color = [0 0 0];
er.LineStyle = 'none';
hold off




