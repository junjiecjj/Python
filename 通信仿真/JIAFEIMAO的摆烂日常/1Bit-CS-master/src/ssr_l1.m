function Xt = ssr_l1(Y, A, par, lambda)

	% Sparse signal recovery via \|x\|_1 minimization based on FISTA
	%
    % By Shuai Huang, The Johns Hopkins University
    % Email: shuang40@jhu.edu 
    % Date: 11/26/2016
	%
	% Y             : the observation/measurement vector
	% A             : the sensing matrix
	% par           : various parameters
	% lambda        : the regularization parameter lambda
	%

	par.fun = 'l1';

	fun_sg = str2func( [par.fun '_sg'] );
	fun_compute = str2func( ['compute_' par.fun] );

	kappa              = par.kappa;             % Lipschitz constant
	tol              = par.tol;             % The convergence tolerance
	maxiter       	 = par.maxiter;         % Maximum number of iterations in the main loop
	innermaxiter     = par.innermaxiter;    % Maximum number of iterations in the inner loop
	                                        % It can be set to a small number, 1 usually suffices
	p            	 = par.p;               % The parameter p, for l1 norm it is 1
	Xt               = par.X0;              % Initialize X
	epsilon          = par.epsilon;         % A small positive number, usually set 1e-12
	

	k_t = 1;
	k_tm1 = 1;

	Xtm1 = Xt;

	G=A'*A;
	C=A'*Y;

	for iter = 1 : maxiter
	
		%fprintf('   Inner iteration %d\n', iter);
		Ut = Xt + ((k_tm1-1)/k_t)*(Xt-Xtm1);
		Gt = Ut - (1/kappa)*2*(G*Ut-C);
		
		w = fun_sg(Ut, p, epsilon);
		Xtp1 = weighted_shrinkage(Gt, lambda/kappa, w);

        cvg_val = norm(Xtp1 - Xt, 'fro') / norm(Xtp1, 'fro');
        %fprintf('Ite: %d\t%d\n', iter, cvg_val)

		if (cvg_val < tol)
			break;
		end
		
		k_tp1 = 0.5*(1+sqrt(1+4*k_t^2)) ;
		
		k_tm1 = k_t ;
		k_t = k_tp1 ;
		
		Xtm1 = Xt ;
		Xt = Xtp1 ;
		
	end

end


% Compute the weighted soft thresholding 
function X = weighted_shrinkage(Z, lambda, w)

	X = max(abs(Z)-lambda*w, 0) .* sign(Z);
end


% Compute the objective \|x\|_1
function y = compute_l1(X, p)

	y=sum(abs(X).^p);
end


% Compute the weights, i.e. first order derivative w.r.t. |x_i|
function w = l1_sg( sigma, p, epsilon )

	w = ones(size(sigma));
end

