function result = coherent_detection(parIn,sample)
%COHERENT_DETECTION 此处显示有关此函数的摘要
%   此处显示详细说明
uR = Ang2SpaMtx(parIn.numBSAntennas);

%% Training-Based JADCE
 
rxRef = sqrt(1/parIn.PsMax).*pagetranspose(sample.rxRef);
actFlagEst = zeros(parIn.numUEs, 1);
% GMMV-AMP algorithm (Structured Sparsity)


%chSpaEst = zeros(parIn.numUEs,parIn.numBSAntennas,parIn.lenSprCodes);
[chSpaEst,lambda,noiseStdVarEst] = AMP_NNSPL(rxRef, pagetranspose(sample.txRef),...
    parIn.dampAMP, parIn.numAMPIters, parIn.tolAMP,0);

chSpaEst = chSpaEst/parIn.PA;

% epsilon_cg = 0.001*max(max(max(abs(chSpaEst))));
% actFlagEst(sum(sum(abs(chSpaEst)>epsilon_cg,3),2)./parIn.numBSAntennas./parIn.lenSprCodes >= 0.5) = 1;        
actFlagEst = sum(sum(lambda>0.1,3),2)/parIn.numBSAntennas/parIn.lenSprCodes>0.3;
idActUEsEst = find(actFlagEst==1);
numActUEsEst = length(idActUEsEst);

chSpaEst = pagetranspose(chSpaEst);
% % DMMV-SP
% [chSpaEst, supp, ~] = DMMV_SP(rxRef, pagetranspose(sample.txRef), parIn.numActUEs);
% % CG-AD
% epsilon_cg = 0.01*max(max(max(abs(chSpaEst))));
% actFlagEst(sum(sum(abs(chSpaEst)>epsilon_cg,3),2)./parIn.numBSAntennas./parIn.lenSprCodes >= 0.5) = 1;        
% actFlagEst(supp) = 1; 

% nvar = parIn.PA^2.*parIn.varNoise;
% [chSpaEst, supp, iter] = DMMV_OMP(rxRef, pagetranspose(sample.txRef), nvar);

% % CG-AD
% epsilon_cg = 0.01*max(max(max(abs(chSpaEst))));
% actFlagEst(sum(sum(abs(chSpaEst)>epsilon_cg,3),2)./parIn.numBSAntennas./parIn.lenSprCodes >= 0.5) = 1;        
% % actFlagEst(supp) = 1; 
% 
% chSpaEst = chSpaEst/parIn.PA;
% chSpaEst = pagetranspose(chSpaEst);
% idActUEsEst = find(actFlagEst==1);
% numActUEsEst = length(idActUEsEst);

% rxRefAng = pagemtimes(rxRef,conj(uR));
% [chAngEst, ~,~] = AMP_NNSPL(rxRefAng, pagetranspose(sample.txRef),...
%     parIn.dampAMP, parIn.numAMPIters, parIn.tolAMP,1);
% chSpaEst = pagemtimes(uR, pagetranspose(chAngEst));

%% Coherent Data Detection
phiDDtilde = zeros(parIn.lenSprCodes*parIn.numBSAntennas,numActUEsEst);
for idxBlockAnt = 1:parIn.numBSAntennas
    chSpaTemp = squeeze(chSpaEst(idxBlockAnt,idActUEsEst,:)).';
    phiDDtilde((idxBlockAnt-1)*parIn.lenSprCodes+1:idxBlockAnt*parIn.lenSprCodes,:) = ...
                sample.sprCodes(:,idActUEsEst).*chSpaTemp;
end
     
rxPldDD = zeros(parIn.lenSprCodes*parIn.numBSAntennas,parIn.numOFDMSymbols);
for idxBlockAnt = 1:parIn.numBSAntennas
    rxPldDD((idxBlockAnt-1)*parIn.lenSprCodes+1:idxBlockAnt*parIn.lenSprCodes,:) = ...
    sample.rxPld(:,:,idxBlockAnt);
end

txPldModEst = zeros(parIn.numUEs,parIn.numOFDMSymbols);
% noisePowerEst = noiseStdVarEst * noiseStdVarEst;
noisePowerEst = parIn.PA^2.*parIn.varNoise;
txPldModActEst = pinv(phiDDtilde'*phiDDtilde + noisePowerEst*eye(numActUEsEst))*phiDDtilde'*rxPldDD;
txPldModEst(idActUEsEst,:) = txPldModActEst;



result.chSpaEst = chSpaEst;
result.txPldModEst = txPldModEst;
result.idActUEsEst = idActUEsEst;
result.actFlagEst = actFlagEst;



