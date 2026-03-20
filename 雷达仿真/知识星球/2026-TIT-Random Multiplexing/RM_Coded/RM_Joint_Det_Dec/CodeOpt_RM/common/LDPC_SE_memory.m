function [v_ext_out, v_post_out, I_memory_out, BER_out, BER_info]= LDPC_SE_memory(v_in, I_memory, ite, dv, b, dc, d, threshold)
%% Note:
% State evolution of the LDPC code? x + n, x --> QPSK point, n --> Gaussian
% noise with variance v_in.
% Input: v_in
% ite:   number of iterations, can be 1
% I_memory: memory stored in the decoder.

%% Input
v_in_b = v_in;               % QPSK variance to bit variance
v_l = 4 ./ v_in_b;           % Bit variance to llr variance

%%
I_EC = I_memory;             % Initialization

dv = dv(:);
b  = b(:).';
dc = dc(:);
d  = d(:).';
I_EC_old = I_EC;
for i=1:ite
    %% Variable node update 1
    tem1 = Jfunc_inv(I_EC);
    I_EV = b * Jfunc( sqrt( (dv - 1).* tem1 .^ 2 + v_l) );

    %% Check node update
    tem2 = Jfunc_inv(1 - I_EV);

    I_EC = 1 - d * Jfunc( sqrt(dc - 1) .* tem2 );
    
    %% Early stop iteration
	bstop = (I_EC==1.0) + (abs(I_EC-I_EC_old)<threshold);
    if(bstop~=0)
		break;
    end
    I_EC_old = I_EC;
end

%% Output 
I_memory_out = I_EC;        % Output 3

tem3 = Jfunc_inv(I_EC);     % extrinsic

v_l_ext_out = ( sqrt(dv) .* tem3 ).^2;   % Extrinsic
v_l_post_out =  v_l +  v_l_ext_out;      % Post

BER_out = 0.5 * erfc(  sqrt( v_l_post_out ) / (2 * sqrt(2)) ); % Variance to BER, Output 4

v_ext_out = 4 ./ v_l_ext_out;      % Output QPSK extrinsic variance, Output 1
v_post_out = 4 ./ v_l_post_out;    % Output QPSK post variance, Output 2

%convert edge distribution to node distribution
sum_tmp = sum(b(:)./dv);
a = b(:)./dv/sum_tmp;
a = a(:).';

%weighted BER & variance
BER_dv  = BER_out(:);
BER_out = a(:).' * BER_out(:);

tem4 = 0.0;
tem5 = 0.0;
for cnt=1:length(dv)
    tem4 = tem4 + a(cnt) * QPSK_transfer_fit(v_ext_out(cnt));
    tem5 = tem5 + a(cnt) * QPSK_transfer_fit(v_post_out(cnt));
end

v_ext_out = tem4;
v_post_out= tem5;

%weighted BER for information bits only, information bits is only count for
%the high dvs
if(length(dv)>1)
rate = 1 - sum(d(:)./dc(:))/sum(b(:)./dv(:));
[dv_sort, idx_sort] = sort(dv, 'descend');
a_sort = a(idx_sort);
a_sort_cum = cumsum(a_sort);
idx_rate = find(a_sort_cum>=rate);
idx_rate = idx_rate(1);
a_sort_rate = [a_sort(1:(idx_rate-1)) rate-a_sort_cum(idx_rate-1)];
a_sort_rate = a_sort_rate ./ sum(a_sort_rate);
BER_info = a_sort_rate(:).' * BER_dv(idx_sort(1:idx_rate)); % Variance to BER, Output 5
else
BER_info = BER_out;
end
%clip output for plotting
v_ext_out(v_ext_out<1e-20)   = 1e-20;
v_post_out(v_post_out<1e-20) = 1e-20;
BER_out(BER_out<1e-20)       = 1e-20;
BER_info(BER_info<1e-20)     = 1e-20;
end
