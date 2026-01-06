function [metrics] = performance_evaluation_coherent(parIn,sample,result)
%PERFORMANCE_EVALUATION_COHERENT 此处显示有关此函数的摘要
%   此处显示详细说明
ADEP = sum(abs(sample.actFlag - result.actFlagEst))/parIn.numUEs;
idCDUEs = intersect(sample.idActUEs,result.idActUEsEst); % Correct detection


    
metrics.ADEP = ADEP;

if (isempty(idCDUEs))
    metrics.BER = 1;
else
    msgNumEst = pskdemod(result.txPldModEst(idCDUEs,:), parIn.modOrder, 0, 'gray');
    corBit = sum(de2bi(reshape(msgNumEst,[],1)) == ...
        de2bi(reshape(sample.txPldNum(idCDUEs,:),[],1)),'all');
    BER = 1- corBit/parIn.bitsperFrame;
    metrics.BER= BER;
end

%chSpaEst = result.chSpaEst;
NMSE_CE = sum(abs(sample.chSpaAct - result.chSpaEst).^2,'all')/...
    sum(abs(sample.chSpaAct.^2),'all');
metrics.NMSE_CE = NMSE_CE;
end