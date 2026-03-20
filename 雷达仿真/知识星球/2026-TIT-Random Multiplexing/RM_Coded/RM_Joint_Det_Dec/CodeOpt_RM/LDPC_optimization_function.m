function [variable_edge_degree, variable_edge_weight, parity_edge_degree, parity_edge_weight, snr_dec_ap, v_dec_ext, v_dec_post, real_R] = LDPC_optimization_function(truncated_v, feedback, converage_theshold, output_path, max_tial, deg_set, snr_in, var_in, variable_edge_degree_init, variable_edge_weight_init, parity_edge_degree, parity_edge_weight, figure_num)
%==========================================================================
% LDPC code optimization for equal-rate equal-power IDMA over AWGN channel
%==========================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%set system parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%converage_theshold = ConvThres; %degree distribution converage threshold. Usually = 0.05.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%set tranfer function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(var_in(1)<var_in(end))
    snr_in = fliplr(snr_in);
    var_in = fliplr(var_in);
end
snr_in = snr_in(:).';
var_in = var_in(:).';

if(feedback=='E')
    snr = [(0.0:0.1:0.9).*snr_in(1), snr_in];
    v   = [ones(1,10), var_in];
elseif(feedback=='P')
    snr = snr_in;
    v   = var_in;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initial degree distributions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
variable_edge_degree = deg_set(:);
variable_edge_weight(variable_edge_degree) = 0;
variable_edge_weight(variable_edge_degree_init) = variable_edge_weight_init;
variable_edge_weight = variable_edge_weight(deg_set);
variable_edge_weight = variable_edge_weight(:);
parity_edge_degree   = parity_edge_degree(:);
parity_edge_weight   = parity_edge_weight(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%optimizing degree distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
trial = 0;
bConverage = 0;
while(bConverage==0)
trial = trial + 1;
display(['**************** ' num2str(trial) ' trail(s) ****************']);

dv = variable_edge_degree;
b  = variable_edge_weight;
dc = parity_edge_degree;
d  = parity_edge_weight;

if(feedback == 'E')
[dv_opt, b_opt, real_R] = LDPC_optimization_ext(truncated_v,snr_in,var_in,dv,b,dc,d,deg_set,output_path);
elseif(feedback == 'P')
[dv_opt, b_opt, real_R] = LDPC_optimization_post(truncated_v,snr_in,var_in,dv,b,dc,d,deg_set,output_path);
end

[snr_dec_ap, v_dec_ext, v_dec_post, ~] = LDPC_transfer(dv_opt,b_opt,dc,d,snr,v,feedback,figure_num);

%check converage
vec_angle = sum(variable_edge_weight(:).*b_opt(:))/(norm(variable_edge_weight)*norm(b_opt));
if((1.0-vec_angle)<converage_theshold) 
	bConverage = 1;
else
	bConverage = 0;
end

%update degree disrtribution.
variable_edge_weight = b_opt;
variable_edge_degree = dv_opt;

dv_opt = dv_opt(b_opt>=0.0001);
b_opt = b_opt(b_opt>=0.0001);
save_degree(b_opt,dv_opt,d,dc,snr,v,snr_dec_ap,v_dec_ext,v_dec_post,output_path);

display(num2str([dv_opt(:), b_opt(:)].'));

display(['Achieved rate = ' num2str([real_R sum(real_R)]*2)]);
toc;

pause(0.1);
if(trial >= max_tial || real_R <= 0.0)
    break;                                                                                                                                                                                                                                                                           
end
end%for while

variable_edge_degree = variable_edge_degree(variable_edge_weight>=0.0001);
variable_edge_weight = variable_edge_weight(variable_edge_weight>=0.0001);

if(bConverage==0)
    warning('Degree Optimization Failed! @ bConverage==0\n');
end
if(real_R <= 0.0)
    error('Degree Optimization Failed! @ real_R <= 0.0\n');
end
save([output_path '\LDPC_Optimization_Main.mat']);
end%for function

function save_degree(lemda,lemda_index,rou,rou_index,snr,v,snr_dec_ap,v_dec_ext,v_dec_post, output_path)
    variable_edge_degree = lemda_index;
    variable_edge_weight = lemda;
    parity_edge_degree   = rou_index;
    parity_edge_weight   = rou;
    save([output_path '/degree.mat'], 'parity_edge_degree', 'parity_edge_weight', 'variable_edge_degree', 'variable_edge_weight', 'snr', 'v', 'snr_dec_ap', 'v_dec_ext', 'v_dec_post');
end
