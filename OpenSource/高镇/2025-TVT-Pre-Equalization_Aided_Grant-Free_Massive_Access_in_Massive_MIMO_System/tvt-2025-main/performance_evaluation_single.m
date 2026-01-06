function [metrics] = performance_evaluation_single(parIn,sample,result)
%PERFORMANCE_EVALUATION_JADD 此处显示有关此函数的摘要
%   此处显示详细说明
metrics.NMSE_CE = zeros(1,parIn.numAlgIters);
metrics.BER = zeros(1,parIn.numAlgIters);

ADEP = sum(abs(sample.actFlag - result.actFlagEst))/parIn.numUEs;
idCDUEs = intersect(sample.idActUEs,result.idActUEsEst);


msgNumEst = pskdemod(result.XModEst(idCDUEs,:), parIn.modOrder, 0, 'gray');
corBit = sum(de2bi(reshape(msgNumEst,[],1)) == ...
    de2bi(reshape(sample.msgNum(idCDUEs,:),[],1)),'all');
BER = 1- corBit/parIn.bitsperFrame;
metrics.BER = BER;

metrics.ADEP = ADEP;
end

