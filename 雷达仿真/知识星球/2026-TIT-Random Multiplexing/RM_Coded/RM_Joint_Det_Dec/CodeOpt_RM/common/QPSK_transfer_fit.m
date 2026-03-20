function v_out = QPSK_transfer_fit(v_in)
%v_in is 1/SNR, 
%for QPSK, SNR = 1/N0, for BPSK, SNR = 1/(N0/2).
%thus, the transfer function is the same for QPSK and BPSK
%v_out is the mmse of QPSK or BPSK.
%y = p1 * x^3 + p2 * x^2 + p3 * x^1 + p4;
	idx = find(v_in<0.0,1);
	if(~isempty(idx))
		error('Input variance < 0.0!');
    end
	
	v_out = zeros(size(v_in));
	
    idx = (v_in>=100.0);
    if(~isempty(idx))% 100.0 <= v_in
        v_out(idx) = 1 - 1 ./ v_in(idx);
    end
    
	idx = (v_in>=10.0 & v_in<100.0);
    if(~isempty(idx))% 10 <= v_in < 100
        p1 = 1.510213727651871e-003;
        p2 = -2.477005243610372e-002;
        p3 = -8.843202798365679e-001;
        p4 = -1.780824384419058e-001;
        log_v_in = log(v_in(idx));	
        y = ((p1 .* log_v_in + p2) .* log_v_in + p3) .* log_v_in + p4;	
        v_out(idx) = 1.0 ./ exp(exp(y));
    end
    
	idx = (v_in<10.0 & v_in>0.0);
    if(~isempty(idx))% v_in < 10        
        p1 = -3.316570358980561e-003;
        p2 = -8.627700132715920e-003;
        p3 = -8.812076898926651e-001;
        p4 = -2.270796619237218e-001;
        log_v_in = log(v_in(idx));
        y = ((p1 .* log_v_in + p2) .* log_v_in + p3) .* log_v_in + p4;        
        v_out(idx) = 1.0 ./ exp(exp(y));
    end
end