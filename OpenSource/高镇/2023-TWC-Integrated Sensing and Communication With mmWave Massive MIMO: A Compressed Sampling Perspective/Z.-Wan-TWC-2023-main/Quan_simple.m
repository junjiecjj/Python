function [Y_quan, QR, QI] = Quan_simple(Y,B,type)
    % Quantize Y
    if(max(abs([reshape((real(Y)),[],1);reshape(imag(Y),[],1)]))>1); error('Normalize the Input Please!'); end
    if(isinf(B))
        Y_quan = Y;
        QR = real(Y);
        QI = imag(Y);
    else
        if(strcmpi(type, 'lloyd'))
            [partition,codebook] = lloyds([reshape((real(Y)),[],1);reshape(imag(Y),[],1)],2^B);
        elseif(strcmpi(type, 'uniform'))
            partition = -1+(1:2^B-1)/2^(B-1);
            codebook  = -1+1/2^B+(0:2^B-1)/2^(B-1);
        else
            error('Unrecognized Input Type.')
        end
        [~,quants_re,~] = quantiz(real(Y),partition,codebook);
        [~,quants_im,~] = quantiz(imag(Y),partition,codebook);
        QR = quants_re.';
        QI = quants_im.';
        Y_quan = QR+1i*QI;
    end
end