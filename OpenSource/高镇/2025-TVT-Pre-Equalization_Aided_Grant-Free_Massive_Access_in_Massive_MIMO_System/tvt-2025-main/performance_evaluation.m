function [metrics] = performance_evaluation(parIn,sample,result)
%PERFORMANCE_EVALUATION 此处显示有关此函数的摘要
%   此处显示详细说明

metrics.NMSE_CE = zeros(1,parIn.numAlgIters);
metrics.BER = zeros(1,parIn.numAlgIters);

ADEP = sum(abs(sample.actFlag - result.actFlagEst))/parIn.numUEs;
idCDUEs = intersect(sample.idActUEs,result.idActUEsEst);

msgNumEst = pskdemod(result.XModEstCoarse(idCDUEs,:), parIn.modOrder, 0, 'gray');
corBit = sum(de2bi(reshape(msgNumEst,[],1)) == ...
    de2bi(reshape(sample.msgNum(idCDUEs,:),[],1)),'all');
BERTmp = 1- corBit/parIn.bitsperFrame;
metrics.BERCoarse = BERTmp ;


for idxIter = 1:parIn.numAlgIters
    XModEst = result.XModEst(:,:,idxIter);

    msgNumEst = pskdemod(XModEst(idCDUEs,:), parIn.modOrder, 0, 'gray');
    corBit = sum(de2bi(reshape(msgNumEst,[],1)) == ...
        de2bi(reshape(sample.msgNum(idCDUEs,:),[],1)),'all');
    BER = 1- corBit/parIn.bitsperFrame;
    metrics.BER(idxIter) = BER;

    if(isfield(result,'chSpaEquEst'))
        chSpaEquEst = result.chSpaEquEst(:,:,:,idxIter);
        % NMSE_CE = sum(abs(sample.chSpaEqu(:,:,idCDUEs) - chSpaEquEst).^2,'all')/...
        %     sum(abs(sample.chSpaEqu(:,:,idCDUEs).^2),'all');
        NMSE_CE = sum(abs(sample.chSpaEqu(:,:,result.idActUEsEst) - chSpaEquEst).^2,'all')/...
        sum(abs(sample.chSpaEqu(:,:,result.idActUEsEst).^2),'all');
        metrics.NMSE_CE(idxIter) = NMSE_CE;
    end
end

metrics.ADEP = ADEP;
end

