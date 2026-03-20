function [mi, vv] = setup_mi_v_transfer_table(dv,b,dc,d,snr_in)
dv= dv(:).';
b = b(:).';
dc= dc(:).';
d = d(:); 
B = b./dv;
B = B/sum(B);

v_l = 4.0 * snr_in;
mi  = 1.0 - [logspace(-16, -2, 1000), logspace(-2, 0, 1000)];
mi  = [0; fliplr(mi(:))];

tem  = Jfunc_inv(1.0-mi) * sqrt(dc-1);
I_EC = 1.0 - Jfunc(tem) * d;
tem  = (Jfunc_inv(I_EC) .^ 2) * dv + v_l;
vv   = QPSK_transfer_fit(4.0 ./ tem) * B(:);

[~, I, ~] = unique(vv, 'first');
mi = mi(I);
vv = vv(I);
if(vv(1)==0.0 && mi(1)~=1.0)
    mi(1) = 1.0;
end
if(vv(1)>0.0 && mi(1)~=1.0)
    mi = [1.0; mi(:)];
    vv = [0.0; vv(:)];
end
end