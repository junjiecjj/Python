function [bits]= QPSK_to_bits(x_s_est,NM)

kk = 1;
for k=1:NM
    if real(x_s_est(k))>0 && imag(x_s_est(k))>0
        x_s(k) = (1+1i)/sqrt(2); y1(kk) = 0; y1(kk+1) = 0;
    else if real(x_s_est(k))>0 && imag(x_s_est(k))<0
            x_s(k) = (1-1i)/sqrt(2); y1(kk) = 0; y1(kk+1) = 1;
    else if real(x_s_est(k))<0 && imag(x_s_est(k))>0
            x_s(k) = (-1+1i)/sqrt(2); y1(kk) = 1; y1(kk+1) = 0;
    else x_s(k) = (-1-1i)/sqrt(2); y1(kk) = 1; y1(kk+1) = 1;
    end
    end
    end
    kk = kk+2;
end
bits = y1;
end