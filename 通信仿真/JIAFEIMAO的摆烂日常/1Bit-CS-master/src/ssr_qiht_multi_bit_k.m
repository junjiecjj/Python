function Xt = ssr_iht(Y, A, par, K)

	% Iterative hard thresholding
	% Make sure the operator norm of A is normalized so that ||A||_2<=1
	% Shuai Huang

	tol              = par.tol;
	maxiter       	 = par.maxiter;

	Xt               = par.X0; % Initialize X
    quant_thd        = par.quant_thd;
    quant_thd_len    = length(quant_thd);
    normalization_cst = par.normalization_cst;

	C=A'*Y;

	for iter = 1 : maxiter

        Xt_tmp = A*Xt;

        % convert Xt_tmp to quantized symbols
        Xt_tmp_quant = zeros(size(Xt_tmp));
        for (i=1:length(Xt_tmp))
            for (j=2:quant_thd_len)
                if ( (Xt_tmp(i)>=quant_thd(j-1))&&(Xt_tmp(i)<quant_thd(j)) )
                    Xt_tmp_quant(i) = 0.5*(quant_thd(j-1)+quant_thd(j));
                    break;
                end
            end
        end

        % just in case the measurements are beyond the quantization range
        for (i=1:length(Xt_tmp))
            if (Xt_tmp(i)<quant_thd(1))
                Xt_tmp_quant(i) = 0.5*(quant_thd(1)+quant_thd(2));
            end
            if (Xt_tmp(i)>=quant_thd(quant_thd_len))
                Xt_tmp_quant(i) = 0.5*(quant_thd(quant_thd_len-1)+quant_thd(quant_thd_len));
            end
        end

        Xt_new = Xt + C - A'*Xt_tmp_quant;

        [Xt_new_abs_sort, sort_idx] = sort(abs(Xt_new), 'descend');
        Xt_new(sort_idx((K+1):end)) = 0;

        cvg_val = norm(Xt_new - Xt) / norm(Xt_new);

        %fprintf('Iter %d:\t%d\n', iter, cvg_val)

		if (cvg_val < tol)
			break;
		end

        Xt = Xt_new;
		
	end

end

