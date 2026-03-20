function [snr_dec_ap, v_dec_ext, v_dec_post, BER_dec, BER_info] = LDPC_transfer(dv,b,dc,d,snr,v,feedback,figure_num)
	snr = sort(snr, 'ascend');
    idx = find(snr>0);
    snr = snr(:).';
	snr_dec_ap = [(0.1:0.1:0.9).*(snr(idx(1))), snr(idx), (1.1:0.1:2).* snr(end)];

	ite = 1000;
	threshold = 1e-6;
	v_in = 1 ./ snr_dec_ap;
	v_dec_ext = zeros(size(v_in));
	v_dec_post= zeros(size(v_in));
	BER_dec = zeros(size(v_in));
	BER_info = zeros(size(v_in));
	for cnt = 1:length(v_in)
		[v_ext, v_post, ~, BER_out, BER_info(cnt)] = LDPC_SE_memory(v_in(cnt), 0, ite, dv, b, dc, d, threshold);
		v_dec_ext(cnt)   = v_ext;
		v_dec_post(cnt)  = v_post;
		BER_dec(cnt)     = BER_out;
		if(v_ext<=2e-20 && v_post <=2e-20 && BER_out <=2e-20)
			break;
		end
    end
    
    if(figure_num>0)
		figure(figure_num);        
		plot(snr, v, 'r-');
		hold on;
		xlabel('\rho');
		ylabel('v');
        plot(snr_dec_ap, v_dec_post, 'm-');  
        legend('DET', 'DEC Post, approx','location','best');
    end

% 	if(figure_num>0)
% 		figure(figure_num);        
%         subplot(2,1,1);
% 		plot(snr, v, 'r-');
% 		hold on;
% 		xlabel('\rho');
% 		ylabel('v');
% 		
%         subplot(2,1,2);
% 		semilogy(10*log10(snr), v, 'r-');
% 		hold on;
% 		xlabel('\rho (dB)');
% 		ylabel('v');
%         if(feedback=='E')    
%             subplot(2,1,1);
%             plot(snr_dec_ap, v_dec_ext, 'g-');
%             legend('DET', 'DEC Ext, approx');
%             subplot(2,1,2);
%             semilogy(10*log10(snr_dec_ap), v_dec_ext, 'g-');
%             legend('DET', 'DEC Ext, approx','location','best');
%         elseif(feedback=='P')
%             subplot(2,1,1);
%             plot(snr_dec_ap, v_dec_post, 'm-');  
%             legend('DET', 'DEC Post, approx','location','best');
%             subplot(2,1,2);
%             semilogy(10*log10(snr_dec_ap), v_dec_post, 'm-');
%             legend('DET', 'DEC Post, approx','location','best');
%         end
% 	end
	
end%for function
