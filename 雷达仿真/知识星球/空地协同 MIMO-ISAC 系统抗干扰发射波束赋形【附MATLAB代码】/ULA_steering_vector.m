function [ a ] = ULA_steering_vector(len,direction)
%  len是向量维度，direction是角度rad
    for ii=1:len
        for jj=1:length(direction)
            a(ii,jj) = exp(1i*pi*(ii-1)*sin(direction(jj)));
        end
    end
end

