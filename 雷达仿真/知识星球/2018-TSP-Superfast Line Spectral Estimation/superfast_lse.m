function output = superfast_lse(y, index, N, varargin)
%SUPERFAST_LSE  Superfast Line Spectral Estimation.
%   output = superfast_lse(y, index, N, ...) performs line spectral estimation
%   based on the method presented in [1].
%
%	[1] T. L. Hansen, B. H. Fleury and B. D. Rao - "Superfast Line Spectral
%	Estimation", submitted to IEEE Transactions on Signal Processing, 2018
%   
%   y is vector with time domain measurements.
%
%   index is a vector of indices where the observations in y are taken.
%   index contains M elements from [1:N] and it must be sorted in ascending order.
%
%   N is the number of reconstructed samples in the observation domain.
%
%   Further arguments can be specified as name,value pairs, e.g. as:
%     superfast_lse(y,index,N, 'verbose', true, ...)
%   Each entry in the 'as' structure specified in the source code can be given this
%   way. If nothing is specified, the default value specified in the source
%   code is used.
%
%   The signal model is
%      y = \sum_k exp( sign * j * 2 * pi * theta_k * (index-1) ) * alpha_k + w.
%   Sign is selected by calling the function with with option (..., 'sign',+/-1).
%   The default is sign = -1. (which is opposite than in the paper).
%	
%	Copyright (c) 2018: Thomas Lundgaard Hansen.
%	This software is being released under the MIT license (see LICENSE file).

	%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Setup struct with default parameters
	% Algorithm settings
	as = struct();
	as.Nit = 500;					% Maximum number of iterations (500 is conservative).
	as.verbose = false;				% Print info during iterations?
	as.plot = false;				% Plot progress in each iteration?
	as.tau_true = nan;				% If not nan, use these as the true values
									% when making plots.
	as.alpha_true = nan;			% Same as tau_true.
	as.sign = -1;					% Sign in the exponential of the signal model.
	as.noiseVar = nan;				% Specify the noise variance. If nan the
									% algorithm will infer it automatically.
									% The algorithm works *much* better with
									% unknown noise var.
	as.pactive = nan;				% If not nan, use this pactive throughout.
	as.thresFactor = 5;				% Heuristic modification of the activation
									% threshold (tau).
	as.snr_cap = 60;				% Cut estimated SNR at approximately 60 dB
									% (can lead to numerical issues otherwise).
	as.noiseVar_bound = 0;			% Do not allow noise variance to go below
									% this value.
	as.L = nan;						% Number of values on grid when searching
									% for components to be activated. If nan
									% use a default.
	as.Kmax = nan;					% Maximum number of sinusoids to estimate.
	as.addremoveMultiple = true;	% Allow multiple components to be activated
									% or deactivated in each iteration.
	as.use_direct = false;			% Use direct matrix inversion in place of
									% superfast/semifast evaluations.
	as.stoppingThres = 1e-7;		% Stop when the objective does not change
									% more than this between two iterations.

	% Add mex to path if needed
	mfilepath=fileparts(which(mfilename));
	if exist('nufft1d')~=3
		addpath(fullfile(mfilepath, 'nufft'));
		if exist('nufft1d')~=3
			error('NUFFT mex files not available on the MATLAB path. Download NUFFT matlab package from http://www.cims.nyu.edu/cmcl/nufft/nufft.html');
		end
	end
	if exist('superfast_lse_utils.m')~=2
		addpath(fullfile(mfilepath, 'utils'));
	end

	% Parse input from varargin
	as = parse_varargin(as, varargin);

	% Open figure if needed
	if as.plot
		num = string2hash(evalc('dbstack()'));
		figure( mod(num, 2^31-2) );
		set(gcf, 'name', mfilename());
	end

	%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Setup
	%%%%%%%%%%%%%%%%%%%%%%%%%
	if isnan(as.L)
		L = 2^round(log2(N*8));
	else
		L = as.L;
	end
	taugrid = linspace(0, 1-1/L, L)';
	M = length(y);
	if isnan(as.Kmax)
		Kmax = M;
	else
		Kmax = as.Kmax;
	end
	% Make sure non-singleton dimension is first
	y = shiftdim(y);
	index = shiftdim(index);

	% Decide on method for computaiton.
	% 1 = superfast O(Nlog^2N) for complete observations
	% 2 = semifast O(K^3 + N^2logN) for subsampled observations
	% 3 = O(LM^2) via direct inversion of C
	assert(all(diff(index)>=1)); % Make sure observation vector is sorted
	if as.use_direct
		method = 3;	% direct inversion
		FFTy = [];
		Agrid = getA(index, as.sign, taugrid);
	elseif M==N
		method = 1;	% superfast
		assert(all(diff(index)==1)); % Make sure observation vector is complete
		Nfft = 2^ceil(log2(2*N - 1));
		FFTy = fft(y, Nfft);
		Agrid = zeros(0, L);	% Used to determine L in subfunctions
	else
		method = 2; % semifast for subsampled observations
		Nfft = 2^ceil(log2(2*N - 1));
		tmp = zeros(N,1);
		tmp(index) = y;
		FFTy = fft(tmp, Nfft);
		Agrid = zeros(0, L);	% Used to determine L in subfunctions
	end

	% Print the utilized method
	if as.verbose
		switch method
		case 1
			fprintf('** Using superfast method. **\n');
			if exist('genschur')~=3 && exist('genschur_recurrence')~=3
				fprintf('Warning: Could not find mex files for running genschur.\n');
				fprintf('For faster runtime, cd to superfast lse folder and run buildmex.m.\n');
			end
		case 2
			fprintf('** Using semifast method. **\n');
		case 3
			fprintf('** Using direct inversion (slow). **\n');
		end
	end

	% Initialize model parameters
	if ~isnan(as.noiseVar)
		beta = as.noiseVar;
	else
		% Assume 1% of energy in y is noise.
		beta = 0.01 * norm(y)^2/M;
	end

	if ~isnan(as.pactive)
		pactive = as.pactive;
	else
		pactive = 0.1;
	end

	% Initialize with empty model
	z = false(Kmax,1);
	tau = nan(Kmax,1);
	gamma = nan(Kmax,1);
	Cinv = precalcCinv(as, y, index, N, FFTy, nan(0,1), nan(0,1), beta, method);
	bfgs_obj = get_bfgs_obj(10);

	%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Iterate
	%%%%%%%%%%%%%%%%%%%%%%%%%
	it = 0;
	obj = inf;
	snr_est = nan;
	extraiters = 0;
	while it < as.Nit
		it = it + 1;
		addedOrRemoved = false;
		if as.verbose
			fprintf('\n\033[4m Iteration %i:', it);
			fprintf('\033[0m\n');
		end

		% Uncomment this to check fast calculations of q,r,s,t,u, etc.
		%verify_vals(as, y, index, N, FFTy, tau, gamma, z, beta, method, ...
			%Agrid, taugrid)

		%%%%%%%%%%%%%%%%%%%%%%%%%
		%% Remove components?
		%%%%%%%%%%%%%%%%%%%%%%%%%
		%[q, s] = calc_qs(as, index, N, Cinv, beta);
		%[z, Cinv, removedAny] = removal_step(z, tau, gamma, beta, pactive, ...
									%Cinv, q, s, as, y, index, N, FFTy, method);
		%addedOrRemoved = addedOrRemoved | removedAny;

		%%%%%%%%%%%%%%%%%%%%%%%%%
		%% Add components?
		%%%%%%%%%%%%%%%%%%%%%%%%%
		if ~exist('lastApproxIter');
			%%%%%%%%%%%%%%%%%%%%%%%%%
			%% Add components via approximate method
			%%%%%%%%%%%%%%%%%%%%%%%%%
			% Evaluate objective at all grid points
			[q, s] = calc_qs_grid(as, y, index, N, Cinv, beta, Agrid);
			if sum(z) == 0
				gammaBar = max(0,max((abs(q).^2 - s) ./ s.^2));
			else
				gammaBar = mean(gamma(z));
			end
			Lgrid = log(1+gammaBar.*s) - abs(q).^2 ./ (1./gammaBar + s) ...
					+ log((1-pactive)/pactive);

			% Find candidates and deactivated indices
			candidates = findpeaks_circular(-Lgrid);	% Candidates
			deactivated = find(~z)';	% Deactivated components
			cand_i = 0;
			deact_i = 0;

			% Loop while there are still candidates and deactivated components
			fMin = inf;
			num_added = 0;
			while cand_i<length(candidates) ...
					& deact_i<length(deactivated) ...
					& num_added < M/10;
				% Calculate gamma, L and mu
				cand_i = cand_i+1;
				l = candidates(cand_i);
				fNew = Lgrid(l);
				activate = abs(q(l))^2/s(l) ...
				  				> ( 1 + 1/(gammaBar*s(l)) ) ...
								* log( (1+gammaBar*s(l)) * ((1-pactive)/pactive) );
				tauNew = taugrid(l);
				dists = abs(tauNew-tau(z));
				dists = min( mod(dists,1), mod(1-dists,1) );
				minDist = min(dists);
				if isempty(minDist)
					minDist = 1;		% No active components
				end
				
				% Check activation criterion
				if abs(q(l))^2/s(l)>1 & activate & fNew<fMin/5 & minDist>0.05/N
					deact_i = deact_i + 1;
					i = deactivated(deact_i);
					z(i) = true;
					tau(i) = taugrid(l);
					gamma(i) = (abs(q(l))^2 - s(l)) / s(l)^2;
					fMin = min(fMin, fNew);
					num_added = num_added + 1;
					if as.verbose
						fprintf(' approx added tau=%.9f (k=%i, gamma=%g)\n', ...
									tau(i), i, gamma(i));
					end
				end
			end
			if num_added > 0
				Cinv = precalcCinv(as, y, index, N, FFTy, tau(z), gamma(z), beta, method);
			end
			if num_added<1 || it>=15
				lastApproxIter = it;
			end
			addedOrRemoved = addedOrRemoved | num_added>0;
		else
			%%%%%%%%%%%%%%%%%%%%%%%%%
			%% Add components via normal method
			%%%%%%%%%%%%%%%%%%%%%%%%%
			for iii = 1:round(M/10)
				% Find inactive component
				i = find(~z,1);
				if ~isempty(i)
					[q, s] = calc_qs_grid(as, y, index, N, Cinv, beta, Agrid);

					% Evaluate objective
					if sum(z) == 0
						gammaBar = max(0,max((abs(q).^2 - s) ./ s.^2));
					else
						gammaBar = mean(gamma(z));
					end
					Lgrid = log(1+gammaBar.*s) - abs(q).^2 ./ (1./gammaBar + s) ...
							+ log((1-pactive)/pactive);
					[~, l] = min(Lgrid);
					activate = abs(q(l))^2/s(l) ...
				  				> ( 1 + 1/(gammaBar*s(l)) ) ...
								* log( (1+gammaBar*s(l)) * ((1-pactive)/pactive) ) ...
								+ as.thresFactor;

					% Check activation criterion
					if abs(q(l))^2/s(l)>1 & activate
						z(i) = true;
						tau(i) = taugrid(l);
						gamma(i) = (abs(q(l))^2 - s(l)) / s(l)^2;
						Cinv = precalcCinv(as, y, index, N, FFTy, tau(z), gamma(z), beta, ...
									method);
						addedOrRemoved = true;
						% Print and update plot
						if as.verbose
							fprintf(' added tau=%.9f (k=%i, gamma=%g)\n', ...
										tau(i), i, gamma(i));
						end
					else
						break;
					end
				else
					break;
				end
				if ~as.addremoveMultiple
					break;
				end
			end
		end

		% If no components in model, break
		if sum(z) == 0
			break;
		end

		%%%%%%%%%%%%%%%%%%%%%%%%%
		%% Update activation probablity
		%%%%%%%%%%%%%%%%%%%%%%%%%
		if isnan(as.pactive)
			pactiveOld = pactive;
			pactive = max(0.1,min(0.5, sum(z)/length(z)));
			if as.verbose
				fprintf(' pactive %.4g -> %.4g\n', pactiveOld, pactive);
			end
		end

		%%%%%%%%%%%%%%%%%%%%%%%%%
		%% Update noise variance
		%%%%%%%%%%%%%%%%%%%%%%%%%
		if isnan(as.noiseVar)
			% Find proposed update
			betaOld = beta;
			[q, s] = calc_qs(as, index, N, Cinv, beta);
			alpha = gamma(z) .* q;
			%Amu = exp(as.sign*j*2*pi*(index-1)*tau(z)') * alpha;
			Amu = calc_Ax(as, index, N, Cinv, alpha);
			trSigmaAhA = beta * sum(gamma(z) .* s);
			M = length(y);
			beta = (norm(y - Amu)^2 + trSigmaAhA) / M;

			% Lower bound on beta is specified by the algorithm user and is
			% application specific.
			beta_min = max( as.noiseVar_bound, mean(abs(y).^2)*10^(-as.snr_cap/10) );
			beta = max(beta, beta_min);

			% Update Cinv
			Cinv = precalcCinv(as, y, index, N, FFTy, tau(z), gamma(z), beta, method);

			% If beta has changed more than 20 %, we invalidate the inverse-Hessian
			% approximation
			if abs( 1 - beta/betaOld ) > 0.2
				bfgs_obj.num_saved = 0;
			end

			% Estimated SNR (used in plots)
			snr_est = (norm(Amu)^2 + trSigmaAhA) / (M*beta);

			% Printf info
			if as.verbose
				fprintf(' noiseVar %.4g -> %.4g (diff=%.2g)\n', ...
							betaOld, beta, beta - betaOld );
			end
		end

		%%%%%%%%%%%%%%%%%%%%%%%%%
		%% Stopping criterion
		%%%%%%%%%%%%%%%%%%%%%%%%%
		% Calculate objective
		objOld = obj;
		[yHCinvy, logDetC] = calc_likelihood(y, index, Cinv, gamma(z), beta);
		obj = logDetC + yHCinvy ...
				- sum(z)*log(pactive) - (length(z)-sum(z))*log(1-pactive+eps());

		% Sanity check
		if isnan(obj)
			warning('Objective is NaN!\n');
		end

		% Print objective
		if as.verbose
			fprintf(' new objective %.9g (diff=%.2g)\n', obj, obj-objOld);
		end

		% Update plot
		update_plot(as, y, index, N, Cinv, z, gamma, tau, beta, snr_est, it, obj, obj-objOld);

		% Check stopping criterion
		if exist('lastApproxIter') && it>lastApproxIter ...
				&& ~addedOrRemoved ...
				&& abs(obj-objOld) < length(y) * as.stoppingThres
			% If maximum number of extraiters have been run, terminate
			if extraiters >= 0
				break;
			else
				% Remove all components which doesnt adhere to the heuristic
				% addition criterion
				[q, s] = calc_qs(as, index, N, Cinv, beta);
				q_tilde = q ./ (1 - gamma(z).*s);
				s_tilde = s ./ (1 - gamma(z).*s);
				gammaBar = mean(gamma(z));
				deactivate = abs(q_tilde).^2./s_tilde ...
				  				< ( 1 + 1./(gammaBar*s_tilde) ) ...
								.* log( (1+gammaBar*s_tilde) * ((1-pactive)/pactive) ) ...
								+ as.thresFactor;
				% See comment in removal_step() regarding instability of Woodburys identity
				deactivate(s_tilde<0) = false;
				if any(deactivate)
					% Remove these components
					z(deactivate) = false;
					Cinv = precalcCinv(as, y, index, N, FFTy, tau(z), gamma(z), beta, method);
					addedOrRemoved = true;
					extraiters = extraiters + 1;
					if as.verbose
						fprintf(' heuristically removed %i components\n', nnz(deactivate));
					end
				end
				if ~any(deactivate) || sum(z)==0
					% No extraiter needed
					break;
				end
			end
		end

		%%%%%%%%%%%%%%%%%%%%%%%%%
		%% Update tau & gamma
		%%%%%%%%%%%%%%%%%%%%%%%%%
		% Setup
		tau_before_update = tau;
		gamma_before_update = gamma;
		innerMaxIters = 5;
		if addedOrRemoved
			% If we have added or removed components invalidate the
			% inverse-Hessian estimate.
			bfgs_obj.num_saved = 0;
		end

		% Iterate
		[deriv, deriv2] = calc_deriv(as, index, N, Cinv, beta, gamma(z));
		for i = 1:innerMaxIters
			% Perform update
			[tau, gamma, beta, Cinv, deriv, deriv2, q, s, bfgs_obj, converged] ...
				= calc_update(z, tau, gamma, beta, Cinv, deriv, deriv2, bfgs_obj, ...
								as, y, index, N, FFTy, method);

			% Check if a component can be deactivated
			[z, Cinv, removedAny] = removal_step(z, tau, gamma, beta, pactive, Cinv, ...
						q, s, as, y, index, N, FFTy, method);
			if removedAny
				bfgs_obj.num_saved = 0;
				[deriv, deriv2] = calc_deriv(as, index, N, Cinv, beta, gamma(z));
			end

			% If we have converged or no components left in model
			if (converged && ~removedAny) || sum(z)==0
				break;
			end
		end

		% Print some info
		if as.verbose
			fprintf('  Summary of (tau,gamma) update:\n');
			fprintf('  tau min change = %.2g, max change = %.2g\n', ...
						min(abs(tau(z)-tau_before_update(z))), ...
						max(abs(tau(z)-tau_before_update(z))) );
			fprintf('  gamma [%.2g, %.2g] -> [%.2g, %.2g], max change = %.2g\n', ...
						min(gamma_before_update(z)), max(gamma_before_update(z)), ...
						min(gamma(z)), max(gamma(z)), ...
						max(abs(gamma(z)-gamma_before_update(z))) );
		end
	end % for-loop

	%%%%%%%%%%%%%%%%%%%%%%%%%
	% Calculate parameters of next component which could have been added (but
	% was not)
	%%%%%%%%%%%%%%%%%%%%%%%%%
	[q, s] = calc_qs_grid(as, y, index, N, Cinv, beta, Agrid);
	if sum(z) == 0
		gammaBar = max(0,max((abs(q).^2 - s) ./ s.^2));
	else
		gammaBar = mean(gamma(z));
	end
	Lgrid = log(1+gammaBar.*s) - abs(q).^2 ./ (1./gammaBar + s) ...
				+ log((1-pactive)/pactive);
	[~, l] = min(Lgrid);
	output.q2 = abs(q(l))^2;
	output.s = s(l);
	output.gammaBar = gammaBar;

	%%%%%%%%%%%%%%%%%%%%%%%%%
	% Get final estimate and make output
	%%%%%%%%%%%%%%%%%%%%%%%%%
	q = calc_qs(as, index, N, Cinv, beta);
	[output.tau, I] = sort(tau(z));
	alpha = gamma(z) .* q;
	output.alpha = alpha(I);
	power = gamma(z);
	output.power = power(I);
	%output.h = exp(as.sign*j*2*pi*[0:N-1]'*output.tau') * output.alpha;
	output.h = calc_Psix(as, N, Cinv, alpha, tau(z));
	output.noiseVar = beta;
	output.pactive = pactive;
	output.iters = it;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper functions related to estimation algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = getA(index, sign_, tau)
	% Opposite sign in the complex exponential, when compared to the paper
	A = exp(sign_*1j*2*pi*(index-1)*tau');
end

function Ax = calc_Ax(as, index, N, Cinv, x)
	if Cinv.method==1 || Cinv.method==2
		Psix = calc_Psix(as, N, Cinv, x);
		Ax = Psix(index);
	else
		Ax = Cinv.A * x;
	end
end

function Psix = calc_Psix(as, N, Cinv, x, tau)
	if Cinv.method==1 || Cinv.method==2
		tmp = conj(Cinv.NUFFTshift) .* x;
		Psix = length(tmp) * nufft1d1(length(tmp), Cinv.NUFFTtau, tmp, ...
						as.sign, Cinv.NUFFTeps, N);
	else
		Psix = exp(as.sign*1j*2*pi*[0:N-1]'*tau') * x;
	end
end

function Cinv = precalcCinv(as, y, index, N, FFTy, tau, gamma, beta, method)
	Cinv.method = method;
	if method==1 || method==2
		% Form shifted version of tau for use with NUFFT
		Cinv.NUFFTtau = mod( tau*2*pi + pi, 2*pi ) - pi;
		Cinv.NUFFTshift = exp(-as.sign*1j*Cinv.NUFFTtau*floor(N/2));
		Cinv.NUFFTeps = 1e-12;	% Precision request
	end
	switch method
	case 1
		% Form first row of C
		% Calculate first column as A*gamma (first row of A is all ones)
		cj = conj(Cinv.NUFFTshift) .* gamma;
		column = length(gamma) * nufft1d1(length(cj), Cinv.NUFFTtau, cj, ...
					as.sign, Cinv.NUFFTeps, N);
		%columnOld = Cinv.A * gamma;
		%assert( max(abs(columnOld-column))<1e-7 )
		row = column';
		row(1) = row(1) + beta;

		% Run Generalized Schur algorithm
		[xi, eta, schur_gamma] = genschur(-row(2:end), row(1:end-1));
		rho = [0; flipud(eta(:))] + [flipud(xi(:)); 0];
		delta = cumprod([real(row(1)); 1-abs(schur_gamma(:)).^2]);	% Form delta

		% Precalc Cinv*y
		Nfft = length(FFTy);
		FFTrho = fft(rho, Nfft);
		FFTflipconjrho = fft(conj(flipud(rho)), Nfft);

		T0hy = ifft( FFTflipconjrho .* FFTy );
		T0hy = T0hy(N+1:2*N);
		T0hy(end) = 0;
		T0T0hy = [0; ifft( FFTrho .* fft(T0hy, Nfft) )];
		T0T0hy = T0T0hy(1:N);

		T1y = ifft( FFTrho .* FFTy );
		T1y = T1y(N:2*N-1);
		T1hT1y = ifft( FFTflipconjrho .* fft(T1y, Nfft) );
		T1hT1y = T1hT1y(1:N);

		% Save to Cinv struct
		Cinv.FFTrho = FFTrho;
		Cinv.PhiHCinvy = (T1hT1y - T0T0hy) / delta(end);
		Cinv.rho = rho;
		Cinv.delta = delta;
	case 2
		% Vector to be (NU) Fourier transformed
		diagvec = zeros(N,1);
		diagvec(index) = 1;	% |\bPhi_m,Mm|^2
		Cinv.AHA = form_PsiHdiagPsi(as, diagvec, Cinv, N);

		% Get Cholesky decomposition of Sigma inverse
		Cinv.R = chol(Cinv.AHA/beta + diag(1./gamma));

		% Precalc SigmaAHA
		Cinv.SigmaAHA = Cinv.R \ (Cinv.R' \ Cinv.AHA);

		% Calc PhiH Cinv y
		PhiHy = zeros(N,1);
		PhiHy(index) = y;
		tmp = Cinv.NUFFTshift .* nufft1d2(length(tau), tau*2*pi, ...
						-as.sign, Cinv.NUFFTeps, N, PhiHy);
		tmp = Cinv.R \ (Cinv.R' \ tmp);		% multiply by Sigma
		tmp = conj(Cinv.NUFFTshift) .* tmp;
		tmp = length(tmp) * nufft1d1(length(tmp), Cinv.NUFFTtau, tmp, ...
						as.sign, Cinv.NUFFTeps, N);
		Cinv.PhiHCinvy = complex(zeros(N,1));
		Cinv.PhiHCinvy(index) = y/beta - tmp(index)/beta^2;
	case 3
		Cinv.A = getA(index, as.sign, tau);
		M = length(y);
		C = beta*eye(M) + Cinv.A * diag(gamma) * Cinv.A';
		[Cinv.U, Cinv.S] = svd(C);
		Cinv.Cinv = Cinv.U * inv(Cinv.S) * Cinv.U';
		Cinv.PhiHCinvy = complex(zeros(N,1));
		Cinv.PhiHCinvy(index) = Cinv.Cinv * y;
	otherwise
		error('invalid method');
	end
end

% Used in semifast case
function out = form_PsiHdiagPsi(as, diagvec, Cinv, N)
	K = length(Cinv.NUFFTtau);

	out = complex(zeros(K,K));
	if K>1
		% Points xj at which the NUFFT is evaluated
		xj = bsxfun(@minus, Cinv.NUFFTtau, Cinv.NUFFTtau');	% differences between all taus
		triu_idx = triu(true(K,K), 1);
		xj = xj(triu_idx);	% strict upper triangular part as vector
		xj = mod( xj + pi, 2*pi ) - pi;

		% Perform NUFFT
		shiftvec = exp(-as.sign*1j*xj*floor(N/2));
		tmp = shiftvec .* nufft1d2(length(xj), xj, ...
						-as.sign, Cinv.NUFFTeps, N, diagvec);

		% Form PsiH diag(diagvec) Psi
		out = complex(zeros(K,K));
		out(triu_idx) = tmp;
	end
	out = out + out';
	out(logical(eye(K))) = sum(diagvec);
end

% Used in semifast case
function out = form_PsiHdiagPsi_grid(as, diagvec, Cinv, N, L)
	K = length(Cinv.NUFFTtau);
	out = complex(zeros(K,L));
	taugrid = linspace(0, 1-1/L, L)';
	taugrid = mod( taugrid*2*pi + pi, 2*pi ) - pi;
	if K>1
		% Points xj at which the NUFFT is evaluated
		xj = bsxfun(@minus, Cinv.NUFFTtau, taugrid');	% differences between all taus
		xj = mod( xj + pi, 2*pi ) - pi;
		xj = reshape(xj, [], 1);

		% Perform NUFFT
		shiftvec = exp(-as.sign*1j*xj*floor(N/2));
		out = shiftvec .* nufft1d2(length(xj), xj, ...
						-as.sign, Cinv.NUFFTeps, N, diagvec);

		% Reshape to get result
		out = reshape(out, K, L);
	end
end

function [q, s] = calc_qs(as, index, N, Cinv, beta)
	% Calculate q
	if Cinv.method==1 || Cinv.method==2
		q = Cinv.NUFFTshift .* nufft1d2(length(Cinv.NUFFTtau), Cinv.NUFFTtau, ...
						-as.sign, Cinv.NUFFTeps, N, Cinv.PhiHCinvy);
	elseif Cinv.method==3
		q = Cinv.A' * Cinv.PhiHCinvy(index);
	end

	% Calculate s
	if nargout>=2
		switch Cinv.method
		case 1
			% Calculate s via omega_s
			omega_s = calc_omega_s(as, index, N, Cinv, beta);
			omega_s(1) = real(omega_s(1) / 2);	% This value should be real
			s = Cinv.NUFFTshift .* nufft1d2(length(Cinv.NUFFTtau), Cinv.NUFFTtau, ...
						-as.sign, Cinv.NUFFTeps, N, conj(omega_s));
			s = 2*real(conj(s));
		case 2
			RHinvAHA = (Cinv.R' \ Cinv.AHA);
			s = real(diag(Cinv.AHA))/beta ...
				- real(sum( Cinv.AHA .* Cinv.SigmaAHA.', 2))/beta^2;
		case 3
			CinvA = Cinv.Cinv * Cinv.A;
			s = real(sum( Cinv.A' .* CinvA.', 2));
		end
	end
end

function [q, s] = calc_qs_grid(as, y, index, N, Cinv, beta, Agrid)
	% Calculate q via FFT
	L = size(Agrid,2);
	if as.sign > 0
		q = fft(Cinv.PhiHCinvy, L);
	else
		q = L * ifft(Cinv.PhiHCinvy, L);
	end

	% Calculate s
	if nargout>=2
		switch Cinv.method
		case 1
			% Calculate s via omega_s
			omega_s = calc_omega_s(as, index, N, Cinv, beta);
			omega_s(1) = real(omega_s(1) / 2);	% This value should be real
			if as.sign < 0
				s = 2*real(fft(omega_s, L));
			else
				s = L*2*real(ifft(omega_s, L));
			end
		case 2
			M = length(y);
			diagvec = zeros(N,1);
			diagvec(index) = 1;	% |\bPhi_m,Mm|^2
			AhAg = form_PsiHdiagPsi_grid(as, diagvec, Cinv, N, L);
			tmp = Cinv.R' \ AhAg;
			s = M / beta - real(sum(abs(tmp).^2,1))' / beta^2;
		case 3
			CinvAgrid = Cinv.Cinv * Agrid;
			s = real(sum( Agrid' .* CinvAgrid.', 2));
		end
	end
end

function omega_s = calc_omega_s(as, index, N, Cinv, beta)
	switch Cinv.method
	case 1
		Nfft = length(Cinv.FFTrho);
		qq = [0:N-1]';
		omega1 = conj(ifft( conj(Cinv.FFTrho) .* Cinv.FFTrho ));
		omega1 = (2-N+qq) .* omega1(1:N);
		tmp = fft(qq.*Cinv.rho, Nfft);
		omega2 = conj(ifft( conj(tmp) .* Cinv.FFTrho ));
		omega2 = 2*omega2(1:N);
		omega_s = (omega1 + omega2) / Cinv.delta(end);
	case 2
		% This code should never be used
		K = length(Cinv.NUFFTtau);
		% First form Sigma
		Sigma = Cinv.R \ (Cinv.R' \ eye(K));

		% Form output of 2 dimensional NUFFT
		[xj,yj] = meshgrid(-Cinv.NUFFTtau, Cinv.NUFFTtau);
		xj = - xj(:); yj = - yj(:);
		cj = exp(-as.sign*1j*floor(N/2)*(xj+yj)) .* Sigma(:);
		PsiSigmaPsiH = K^2 * nufft2d1(K^2, yj, xj, cj, ...
			-as.sign, Cinv.NUFFTeps, N, N);

		% Form PhiH Cinv Phi
		PhiHCinvPhi = complex(zeros(N));
		PhiHCinvPhi(index, index) ...
			= eye(length(index)) / beta ...
				- PsiSigmaPsiH(index, index) / beta^2;

		% Get omega_s(i) for i = 0,...,N-1
		omega_s = complex(zeros(N,1));
		for i = 0:N-1
			omega_s(i+1) = sum(diag(PhiHCinvPhi, i));
		end
	end
end

function [r, t, u, v, x] = calc_rtuvx(as, index, N, Cinv, beta)
	% Calculate r
	if Cinv.method==1 || Cinv.method==2
		diagvec = zeros(N,1);
		diagvec(index) = as.sign * 2*pi*(index-1);
		fk = diagvec .* Cinv.PhiHCinvy;
		r = Cinv.NUFFTshift .* nufft1d2(length(Cinv.NUFFTtau), Cinv.NUFFTtau, ...
						-as.sign, Cinv.NUFFTeps, N, fk);
	elseif Cinv.method==3
		r = Cinv.A' * (as.sign*2*pi*(index-1) .* Cinv.PhiHCinvy(index) );
	end

	% Calculate t
	if nargout>=2
		switch Cinv.method
		case 1
			Nfft = length(Cinv.FFTrho);
			qq = [0:N-1]';

			% Find omega_t
			tmp1 = fft(qq.*Cinv.rho, Nfft);
			tmp2 = fft([zeros(N-1,1); qq.*Cinv.rho], Nfft);
			omega1 = conj(ifft( conj(tmp1) .* tmp2 ));
			omega1 = -omega1(1:2*N-1);
			FFTrhoPadded = fft([zeros(N-1,1); Cinv.rho], Nfft);
			tmp = fft( (qq.^2 + (N-1)*(qq-(N-2)/2)).*Cinv.rho, Nfft);
			omega2 = conj(ifft( conj(tmp) .* FFTrhoPadded ));
			omega2 = omega2(1:2*N-1);
			ii = (-(N-1):(N-1))';
			omega3 = conj(ifft( conj(Cinv.FFTrho) .* FFTrhoPadded ));
			omega3 = ii .* ( N - (3+ii)/2 ) .* omega3(1:2*N-1);
			omega_t = 2 * pi * (omega1 + omega2 + omega3) / Cinv.delta(end);

			% Calculate t from omega_t
			omega_t(N) = real(omega_t(N)/2);
			t1 = Cinv.NUFFTshift .* nufft1d2(length(Cinv.NUFFTtau), Cinv.NUFFTtau, ...
						-as.sign, Cinv.NUFFTeps, N, conj(omega_t(N:end)));
			t2 = Cinv.NUFFTshift .* nufft1d2(length(Cinv.NUFFTtau), Cinv.NUFFTtau, ...
						-as.sign, Cinv.NUFFTeps, N, flipud(omega_t(1:N)));
			t = as.sign * (conj(t1) + t2);
		case 2
			Dmtx = form_PsiHdiagPsi(as, diagvec, Cinv, N);
			t = real(diag(Dmtx)) / beta ...
				- sum( Dmtx .* Cinv.SigmaAHA.', 2) / beta^2;
		case 3
			CinvA = Cinv.Cinv * Cinv.A;
			t = sum( Cinv.A' .* (as.sign*2*pi*diag(index-1)*CinvA).', 2);
		end
	end

	% Calculate u, v, x
	if nargout>=3
		% Calculate u
		if Cinv.method==1 || Cinv.method==2
			fk = diagvec.^2 .* Cinv.PhiHCinvy;
			u = Cinv.NUFFTshift .* nufft1d2(length(Cinv.NUFFTtau), Cinv.NUFFTtau, ...
						-as.sign, Cinv.NUFFTeps, N, fk);
		elseif Cinv.method==3
			u = Cinv.A' * ((2*pi*diag(index-1)).^2 * Cinv.PhiHCinvy(index));
		end

		% Calculate v and x
		switch Cinv.method
		case 1
			% Find omega_x
			tmp1 = fft(qq.*Cinv.rho, Nfft);
			tmp = fft( ((2-N)*qq + qq.^2).*Cinv.rho, Nfft);
			omega1 = conj(ifft( conj(tmp) .* tmp1 ));
			omega1 = omega1(1:N);
			tmp = fft( (-qq.^3/3 + (N^2-3*N+7/3)*qq).*Cinv.rho, Nfft);
			omega2 = conj(ifft( conj(tmp) .* Cinv.FFTrho ));
			omega2 = omega2(1:N);
			omega3 = conj(ifft( conj(Cinv.FFTrho) .* Cinv.FFTrho ));
			omega3 = ( N^2/2*qq - 3/2*N*qq + 7/6*qq - qq.^3/6 ...
						- 1/3*N^3 + 3/2*N^2 - 13/6*N + 1 ) .* omega3(1:N);
			omega_x = 4*pi^2 * (omega1 + omega2 + omega3) / Cinv.delta(end);

			% Calculate x from omega_x
			omega_x(1) = real(omega_x(1) / 2);	% This value should be real
			x = Cinv.NUFFTshift .* nufft1d2(length(Cinv.NUFFTtau), Cinv.NUFFTtau, ...
						-as.sign, Cinv.NUFFTeps, N, conj(omega_x));
			x = 2*real(conj(x));

			% Find omega_v
			tmp2 = fft(qq.^2.*Cinv.rho, Nfft);
			tmp1padded = fft([zeros(N-1,1); qq.*Cinv.rho], Nfft);
			omega1 = conj(ifft( conj(tmp2) .* tmp1padded ));
			omega1 = flipud(conj(omega1(1:2*N-1))) - omega1(1:2*N-1);
			tmp1 = fft(qq.*Cinv.rho, Nfft);
			omega2 = conj(ifft( conj(tmp1) .* tmp1padded ));
			omega2 = (3-2*N)*omega2(1:2*N-1);
			tmp = fft( (2/3*qq.^3 + (N-1)*qq.^2 + (N^2-3*N+7/3)*qq).*Cinv.rho, Nfft);
			FFTrhoPadded = fft([zeros(N-1,1); Cinv.rho], Nfft);
			omega3 = conj(ifft( conj(tmp) .* FFTrhoPadded ));
			omega3 = omega3(1:2*N-1);
			ii = (-(N-1):(N-1))';
			omega4 = conj(ifft( conj(Cinv.FFTrho) .* FFTrhoPadded ));
			omega4 = ( 3/2*(ii-N).^2 + 1/3*(ii.^3-N^3) + N*ii.*(N-ii) ...
						+ 13/6*(ii-N) + 1) .* omega4(1:2*N-1);
			omega_v = 4*pi^2 * (omega1 + omega2 + omega3 + omega4) / Cinv.delta(end);

			% Calculate v from omega_v
			omega_v(N) = real(omega_v(N)/2);	% This value should be real
			v1 = Cinv.NUFFTshift .* nufft1d2(length(Cinv.NUFFTtau), Cinv.NUFFTtau, ...
						-as.sign, Cinv.NUFFTeps, N, conj(omega_v(N:end)));
			v2 = Cinv.NUFFTshift .* nufft1d2(length(Cinv.NUFFTtau), Cinv.NUFFTtau, ...
						-as.sign, Cinv.NUFFTeps, N, flipud(omega_v(1:N)));
			v = conj(v1) + v2;
		case 2
			% Calculate v
			D2mtx = form_PsiHdiagPsi(as, diagvec.^2, Cinv, N);
			v = real(diag(D2mtx)) / beta ...
				- sum( D2mtx .* Cinv.SigmaAHA.', 2) / beta^2;

			% Calculate x
			SigmaDmtx = Cinv.R \ (Cinv.R' \ Dmtx);
			x = real(diag(D2mtx)) / beta ...
				- sum( Dmtx .* SigmaDmtx.', 2) / beta^2;
		case 3
			v = sum( Cinv.A' .* ((2*pi*diag(index-1)).^2 * CinvA).', 2);
			DA = - as.sign * 2*pi*diag(index-1) * Cinv.A;
			x = sum( DA' .* (Cinv.Cinv*DA).', 2);
		end
	end
end

function [yHCinvy, logDetC] = calc_likelihood(y, index, Cinv, gamma, beta)
	yHCinvy = real(y' * Cinv.PhiHCinvy(index));
	switch Cinv.method
	case 1
		logDetC = sum(log(Cinv.delta));
	case 2
		logDetC = length(y)*log(beta) ...
					+ sum(log(gamma)) ...
					+ 2*sum(log(diag(Cinv.R)));
	case 3
		logDetC = sum(log(diag(Cinv.S)));
	end
end

function [deriv, deriv2, q, s] = calc_deriv(as, index, N, Cinv, beta, gamma)
	[q, s] = calc_qs(as, index, N, Cinv, beta);
	[r, t, u, v, x] = calc_rtuvx(as, index, N, Cinv, beta);

	% Here the sign of terms which involve D has been switched, because of the
	% change of sign in the exp() of the signal model
	deriv_tau = 2 * gamma .* imag(t - conj(q).*r);
	deriv2_tau = 2 * gamma .* real(x - v + gamma.*t.^2 - gamma.*x.*s ...
				-2*gamma.*t.*r.*conj(q) + gamma.*x.*abs(q).^2 ...
				- abs(r).^2 + gamma.*s.*abs(r).^2 + u.*conj(q));

	deriv_gamma = s - abs(q).^2;
	deriv2_gamma = 2*abs(q).^2 .* s - s.^2;

	deriv = [deriv_tau; deriv_gamma];
	deriv2 = [deriv2_tau; deriv2_gamma];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Update functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z, Cinv, removedAny] ...
		= removal_step(z, tau, gamma, beta, pactive, Cinv, q, s, ...
						as, y, index, N, FFTy, method)
	removedAny = false;
	while true
		% Search for component to remove
		q_tilde = q ./ (1 - gamma(z).*s);
		s_tilde = s ./ (1 - gamma(z).*s);
		Ldiff = abs(q_tilde).^2./s_tilde ...
						- ( 1 + 1./(gamma(z).*s_tilde) ) ...
						.* log( (1+gamma(z).*s_tilde) * ((1-pactive)/pactive) );

		% s_tilde may become negative in high SNR due to numerical
		% instability of Woodburys matrix identity, as applied to
		% obtain s_tilde and q_tilde. In this case the objective becomes
		% complex-valued. We have observed that the
		% component should always be kept in the model for those
		% components where this instability occurs.
		Ldiff(s_tilde<0) = inf;
		if any(Ldiff<0)
			% Remove only a single component
			[~, i] = min(Ldiff);
			zidx = find(z);
			assert(z(zidx(i)));
			z(zidx(i)) = false;
			Cinv = precalcCinv(as, y, index, N, FFTy, tau(z), gamma(z), beta, method);
			removedAny = true;
			% Print and update plot
			if as.verbose
				fprintf(' removed tau=%.9f (k=%i, gamma=%.2g)\n', ...
							tau(i), zidx(i), gamma(i));
			end
		else
			break
		end

		if ~as.addremoveMultiple
			break;
		end

		% For next iteration
		[q, s] = calc_qs(as, index, N, Cinv, beta);
	end
end

function [tau, gamma, beta, Cinv, deriv, deriv2, q, s, bfgs_obj, converged] ...
		= calc_update(z, tau, gamma, beta, Cinv, deriv, deriv2, bfgs_obj, ...
						as, y, index, N, FFTy, method)
	invCount = 0;
	K = sum(z);
	M = length(y);
	converged = false;

	% Scaling
	% This scaling of the (tau,gamma) variables makes sure that the rate of
	% change in the objective function is on the same order of magnitude for
	% all variables.
	% The scaling S is defined by the variable substitution x = y * S, such
	% that we work on g(y) = f(y*S) instead of f(x). Then
	% g' = f'(x) * S
	% g'' = f''(x) * S^2
	% The scaling has been selected such that |g''| \approx 1
	scaling = [ones(K,1)*0.02/N; gamma(z)];
	% Those coordinates for which the diagonal Hessian of the rescaled problem
	% is larger than some small positive constant (i.e. effectively positive)
	idx = deriv2 .* (scaling.^2) > 1e-5;
	% Get initial inverse-Hessian approximation (it is diagonal)
	initial_diag_H = 1 ./ scaling.^2;		% Corresponds to g''=1 in scaled domain
	initial_diag_H(idx) = deriv2(idx);

	if ~all(idx)
		% Some of the diagonal entries of the Hessian was negative (or very
		% small).
		% Thus the Hessian is (nearly) not positive-definite, and a
		% quasi-Newton method doesnt make much sense. We will therefore only
		% use the heuristic initial inverse-Hessian in calculating the search
		% direction.
		bfgs_obj.num_saved = 0;
	end

	% Get search direction
	delta = get_bfgs_direction(bfgs_obj, initial_diag_H, deriv);

	% Get apprixmate Newton decrement
	nt_decr = - deriv' * delta;
	% nt_decr/2 gives the decrease in the 2nd order approximation of the
	% objective function.
	% We break if this is significantly below the reduction required for
	% outer loop converge, to avoid line searches with very small
	% steplengths (those are costly due to large number of function
	% evalutations in the backtracking line search).
	if nt_decr/2 < 1e-1 * length(y) * as.stoppingThres
		if as.verbose
			fprintf(' (tau,gamma) update has converged, Newton decrement = %.2g\n', ...
				nt_decr);
		end
		converged = true;
		[q, s] = calc_qs(as, index, N, Cinv, beta);
		return;
	end

	% Prepare line search
	assert(deriv' * delta < 0);
	deltaTau = delta(1:K);
	deltaGamma = delta(K+1:end);
	tauOld = tau(z);
	gammaOld = gamma(z);
	[yHCinvy, logDetC] = calc_likelihood(y, index, Cinv, gamma(z), beta);
	objOld = logDetC + yHCinvy;

	% Find the maximum stepsize we allow
	% Make sure gamma cannot become zero or negative
	idx = deltaGamma < 0;
	if sum(idx)==0
		max_stepsize = 1;
	else
		max_stepsize = min(-gammaOld(idx)./deltaGamma(idx));
		% Avoid numerical issues (i.e. do not go all the way to gamma=0)
		max_stepsize = (1 - 1e-3) * max_stepsize;
	end
	stepsize = min([1, max_stepsize]);

	% Backtracking line search
	i = 0;
	while true
		i = i + 1;

		% Get new point
		tau(z) = mod(tauOld + stepsize * deltaTau, 1);
		gamma(z) = gammaOld + stepsize * deltaGamma;

		% Calculate objective
		Cinv = precalcCinv(as, y, index, N, FFTy, tau(z), gamma(z), beta, method);
		[yHCinvy, logDetC] = calc_likelihood(y, index, Cinv, gamma(z), beta);
		invCount = invCount + 1;
		objNew = logDetC + yHCinvy;

		% Check Armijo condition
		if objNew <= objOld + 0.05*stepsize*delta'*deriv
			break;
		end

		% Check for small stepsize
		if all(abs(stepsize*deltaTau)<1e-15/N) ...
			&& all(abs(stepsize*deltaGamma)<1e-15*gamma(z))
			%if as.verbose
				fprintf(' could not find an acceptable stepsize, assuming we have converged\n');
				fprintf(' (i=%i, stepsize=%.2g)\n\n', i, stepsize);
			%end
			tau(z) = tauOld;
			gamma(z) = gammaOld;
			Cinv = precalcCinv(as, y, index, N, FFTy, tau(z), gamma(z), beta, method);
			converged = true;
			[q, s] = calc_qs(as, index, N, Cinv, beta);
			return;
		else
			% Reduce stepsize
			rho = max(0.1, 0.5^i);
			stepsize = stepsize * rho;
		end
	end
	assert(all(gamma(z)>=0))

	% Get new derivatives
	deriv_old = deriv;
	deriv2_old = deriv2;
	[deriv, deriv2, q, s] = calc_deriv(as, index, N, Cinv, beta, gamma(z));

	% Update L-BFGS inverse-Hessian estimate
	bfgs_saved = bfgs_obj.num_saved;
	xx = [tau(z); gamma(z)] - [tauOld; gammaOld];
	yy = deriv - deriv_old;
	xxHyy = xx' * yy;
	if xxHyy>0 && all(deriv2_old>0) && all(deriv2>0)
		bfgs_obj = add_bfgs_point(bfgs_obj, xx, yy);
		%Byy = BFGS_B * yy;
		%ByyxxH = Byy * xx';
		%BFGS_B = BFGS_B ...
			%+ (((xxHyy + yy'*Byy) / (xxHyy)^2) * xx ) * xx' ...
			%- 1/(xxHyy) * (ByyxxH + ByyxxH');
	else
		bfgs_obj.num_saved = 0;
	end

	% Print some info
	if as.verbose
		fprintf(' updated (tau,gamma) with stepsize %.2g, invCount=%i, bfgs_saved=%i\n', ...
				stepsize, invCount, bfgs_saved);
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Misc helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function idx = findpeaks_circular(obj)
	% Find peaks
	[~, idx] = findpeaks(obj);

	% Check if first point should also be included
	if obj(1)>obj(2) && obj(1)>obj(end)
		idx = [1; idx];
	end

	% Check if last point should also be included
	if obj(end)>obj(end-1) && obj(end)>obj(1)
		idx = [length(obj); idx];
	end

	[~, I] = sort(obj(idx), 'descend');
	idx = idx(I);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_plot(as, y, index, N, Cinv, z, gamma, tau, beta, snr_est, iter, obj, objDiff)
	if ~as.plot
		return;
	end
	
	% Calculate estimate of alpha (mu)
	q = calc_qs(as, index, N, Cinv, beta);
	mu = gamma(z) .* q;
	tau_est = mod(tau(z)+0.5, 1) - 0.5;

	% Plot in frequency domain
	subplot(3,1,1);
	hold off
	if ~isnan(as.tau_true) & ~isnan(as.alpha_true)
		tau_true = mod(as.tau_true+0.5, 1) - 0.5;
		stem(tau_true, abs(as.alpha_true), 'ro');
		hold on;
	end
	h = stem(tau_est, abs(mu), 'kx');
	set(gca,'yscal','log');
	set(get(h,'BaseLine'),'BaseValue',1e-2);
	ylim([1e-6,2e0]);
	xlim([-0.55,0.55]);

	% Give informative title
	title(sprintf('Iter = %i, Est. SNR = %.1f dB, obj=%g, objDiff=%g', ...
				iter, 10*log10(snr_est), obj, objDiff ));

	% Plot time domain magnitude
	y_est = getA(index, as.sign, tau(z)) * mu;
	subplot(3,1,2);
	hold off
	semilogy(abs(y), 'r', 'displayname','Observed');
	hold on;
	semilogy(abs(y_est), 'b--', 'displayname','Estimated');
	legend show;

	% Plot time domain phase
	subplot(3,1,3);
	hold off
	plot(angle(y), 'r', 'displayname','Observed');
	hold on;
	plot(angle(y_est), 'b--', 'displayname','Estimated');
	legend show;

	drawnow
end

function plot_one_gamma(Cinv, index, N, gamma, z, beta, k)
	zidx = find(z);
	assert(z(zidx(k)));
	i = z(zidx(k));
	[q, s] = calc_qs(as, index, N, Cinv, beta);
	q_tilde = q ./ (1 - gamma(z).*s);
	s_tilde = s ./ (1 - gamma(z).*s);
	ga = linspace(0.1,20,200);
	ff = log(1./ga + s_tilde(k)) + log(ga) ...
			- abs(q_tilde(k))^2/beta ./ (1./ga + s_tilde(k));
	figure(2);
	plot(ga, ff)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% L-BFGS functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function obj = get_bfgs_obj(max_saved)
	obj = struct();
	obj.max_saved = max_saved;
	obj.num_saved = 0;
	obj.newest = 1;
	obj.xx = nan();
	obj.yy = nan();
end

function obj = add_bfgs_point(obj, xx, yy)
	obj.newest = mod( (obj.newest-1) - 1, obj.max_saved ) + 1;
	if obj.num_saved == 0
		K = length(xx);
		obj.xx = nan(K, obj.max_saved);
		obj.yy = nan(K, obj.max_saved);
	end
	obj.num_saved = min(obj.num_saved+1, obj.max_saved);
	obj.xx(:, obj.newest) = xx;
	obj.yy(:, obj.newest) = yy;
end

function delta = get_bfgs_direction(obj, initial_diag_H, deriv)
	qq = deriv;
	k = obj.newest - 1;	% index of current set of saved vectors
	alpha = nan(obj.num_saved,1);
	rho = 1 ./ sum(obj.yy .* obj.xx, 1)';
	for i = 1:obj.num_saved
		k = mod( (k+1) - 1, obj.max_saved ) + 1;
		alpha(i) = rho(k) * (obj.xx(:,k)'*qq);
		qq = qq - alpha(i) * obj.yy(:,k);
	end
	zz = qq ./ initial_diag_H;
	for i = obj.num_saved:-1:1
		beta = rho(k) * (obj.yy(:,k)'*zz);
		zz = zz + obj.xx(:,k) * (alpha(i)-beta);
		k = mod( (k-1) - 1, obj.max_saved ) + 1;
	end
	delta = - zz;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Verification of fast methods vs direct calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function verify_vals(as, y, index, N, FFTy, tau, gamma, z, beta, method, ...
			Agrid, taugrid)
	K = sum(z);
	mu = randn(K,2) * [1;1j];

	% Calculate values using specified method
	fprintf('\n\nCalculating values using method = %i\n', method);
	Cinv = precalcCinv(as, y, index, N, FFTy, tau(z), gamma(z), beta, method);
	[q, s] = calc_qs(as, index, N, Cinv, beta);
	[r, t, u, v, x] = calc_rtuvx(as, index, N, Cinv, beta);
	[q_grid, s_grid] = calc_qs_grid(as, y, index, N, Cinv, beta, Agrid);
	[yHCinvy, logDetC] = calc_likelihood(y, index, Cinv, gamma(z), beta);
	Amu = calc_Ax(as, index, N, Cinv, mu);
	%aa=Cinv.PhiHCinvy;

	% Calculate values using direct method (method = 3)
	fprintf('Calculating values using method = 3\n');
	Agrid = getA(index, as.sign, taugrid);
	Cinv = precalcCinv(as, y, index, N, FFTy, tau(z), gamma(z), beta, 3);
	[qq, ss] = calc_qs(as, index, N, Cinv, beta);
	[rr, tt, uu, vv, xx] = calc_rtuvx(as, index, N, Cinv, beta);
	[qq_grid, ss_grid] = calc_qs_grid(as, y, index, N, Cinv, beta, Agrid);
	[yyHCinvy, llogDetC] = calc_likelihood(y, index, Cinv, gamma(z), beta);
	AAmu = calc_Ax(as, index, N, Cinv, mu);
	%aa-Cinv.PhiHCinvy

	% Compare results
	fprintf('max(abs(q-qq)) = %.4g\n', max(abs(q-qq)));
	fprintf('max(abs(r-rr)) = %.4g\n', max(abs(r-rr)));
	fprintf('max(abs(s-ss)) = %.4g\n', max(abs(s-ss)));
	fprintf('max(abs(t-tt)) = %.4g\n', max(abs(t-tt)));
	fprintf('max(abs(u-uu)) = %.4g\n', max(abs(u-uu)));
	fprintf('max(abs(v-vv)) = %.4g\n', max(abs(v-vv)));
	fprintf('max(abs(x-xx)) = %.4g\n', max(abs(x-xx)));
	fprintf('max(abs(q_grid-qq_grid)) = %.4g\n', max(abs(q_grid-qq_grid)));
	fprintf('max(abs(s_grid-ss_grid)) = %.4g\n', max(abs(s_grid-ss_grid)));
	fprintf('max(abs(yHCinvy-yyHCinvy)) = %.4g\n', max(abs(yHCinvy-yyHCinvy)));
	fprintf('max(abs(logDetC-llogDetC)) = %.4g\n', max(abs(logDetC-llogDetC)));
	fprintf('max(abs(Amu-AAmu)) = %.4g\n', max(abs(Amu-AAmu)));
end

