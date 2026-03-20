 function [dv_opt,b_opt,coding_rate] = LDPC_optimization_post(truncated_v, snr_dec_ap, v_dec_post, dv, dv_weight, dc, dc_weight, deg_set, output_path)
deg_1        = deg_set(:).'-1;
sqrt_dc_1    = sqrt(dc(:).' - 1);
dc_weight    = dc_weight(:);

%-----------------------------------------
% Find A and b for the linear programing 
%-----------------------------------------
quant_num = 32;
count = 1;
for i = 1:length(snr_dec_ap)

	% setup mi -- v_ext transfer table
	[MI, v_in] = setup_mi_v_transfer_table(dv,dv_weight,dc,dc_weight, snr_dec_ap(i));
    %if(v_dec_post(i) > max(v_in))
    %    continue;
    %end
    
    I_EV_min = Jfunc(2 .* sqrt(snr_dec_ap(i)));
    I_EV_max = interp1(v_in, MI, v_dec_post(i));    
    if(v_dec_post(i)<=truncated_v)
        I_EV_max = 1.0;
    end
    if(I_EV_max<I_EV_min || isnan(I_EV_max))
        I_EV_max = I_EV_min;
    end
    
    I_EV = linspace(I_EV_min,I_EV_max,quant_num);
    I_EV = unique(I_EV);    
	if(~isempty(I_EV))
		I_EV = I_EV(:);
		end_idx = count + length(I_EV) - 1;
		
		%% Check node update
		tem2 = Jfunc_inv(1 - I_EV);
		I_EC = 1 - Jfunc( tem2 * sqrt_dc_1 ) * dc_weight;
			
		%% Variable node update
		snr4 = 4*snr_dec_ap(i);
		tem1 = Jfunc_inv(I_EC) .^ 2;
		A(count:end_idx,:) = Jfunc( sqrt(tem1 * deg_1 + snr4) );
		b(count:end_idx,1) = I_EV;
		count = end_idx+1;
	end
end

% linear programming
deg_set_size = length(deg_set);

f = -(deg_set)'.^-1;      % the weighting coefficients of the linear objective

Aeq = ones(1,deg_set_size);
beq = 1;

lb = zeros(deg_set_size,1);
%ub = [];
ub = ones(deg_set_size,1);

%----------------------------------------------------------------
% linprog solves the following problem
%               min     f^T * x
%               s.t.    A * x <= b
%                       A_eq * x == b_eq
%                       lb <= x <= ub
%----------------------------------------------------------------
[x,fval] = linprog(f,-A,-b,Aeq,beq,lb,ub);
fval=fval
dl = -1/fval;
dr = 1/sum(dc_weight./dc);
coding_rate = 1-dl/dr;
display(['real_R = ' num2str(coding_rate)]);

%index = find(x>0.0001);
b_opt = x;%(index);
dv_opt = deg_set;%(index);

save([output_path '/LDPC_Optimization.mat']);
return