function [result] = mimo_detection_omp(parIn,sample)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Proposed MIMO detection scheme 
% 
% Coded by: Yueqing Wang, Beijing Institute of Technology
% Email: 17710820190@163.com
% Last change: 02/11/23
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Coarse Detection
%(norm(sample.YModNoiseEta - sample.sprCodes*sample.XMod,'fro').^2)/numel(sample.YModNoiseEta)
actFlagEst = zeros(parIn.numUEs,1);
[XModEst, index] = SOMP_noise(sample.sprCodes, sample.YModNoiseEta, (1/parIn.zeta)*parIn.varNoise);
actFlagEst(index) = 1;
numActUEsEst = sum(actFlagEst);
idActUEsEst = find(actFlagEst == 1);
result.XModEstCoarse = XModEst;

% [XModEst, lambda] = MMV_AMP_MPSK(sample.YModNoiseEta, sample.sprCodes, ...
% parIn.dampAMP, 50, 1e-4, 0,parIn.modOrder);
% actFlagEst(mean(lambda,2) > parIn.adThOAMP) = 1;
% numActUEsEst = sum(actFlagEst);
% idActUEsEst = find(actFlagEst == 1);
% result.XModEstCoarse = XModEst;

result.XModEst = zeros(parIn.numUEs,parIn.numOFDMSymbols,parIn.numAlgIters);
result.chSpaEquEst = zeros(parIn.numBSAntennas,parIn.lenSprCodes,numActUEsEst,parIn.numAlgIters);


%% Channel Estimation

uR = Ang2SpaMtx(parIn.numBSAntennas);
YModCE = permute(sample.YModNoise,[2,3,1]);
RModCE = pagemtimes(YModCE,conj(uR));

for idxIter = 1:parIn.numAlgIters
phiCEtilde =  permute(repmat(sample.sprCodes(:,idActUEsEst),[1,1,parIn.numOFDMSymbols]),[3,2,1]) .*...
              permute(repmat(XModEst(idActUEsEst,:),[1,1,parIn.lenSprCodes]),[2,1,3]);

[chAngEquEst,~,noiseStdVarEst] = AMP_NNSPL(RModCE,phiCEtilde, parIn.dampAMP, ...
                                   parIn.numAMPIters, parIn.tolAMP, parIn.sparseStr);

chSpaEquEst = pagemtimes(chAngEquEst,uR.');
chSpaEquEst = permute(chSpaEquEst,[2,3,1]);

%% Data Detection   
phiDDtilde = zeros(parIn.lenSprCodes*parIn.numBSAntennas,numActUEsEst);
for idxBlockAnt = 1:parIn.numBSAntennas
    chEquTemp = squeeze(chSpaEquEst(idxBlockAnt,:,:));
    phiDDtilde((idxBlockAnt-1)*parIn.lenSprCodes+1:idxBlockAnt*parIn.lenSprCodes,:) = ...
                sample.sprCodes(:,idActUEsEst).*chEquTemp;
end
     
YModDD = zeros(parIn.lenSprCodes*parIn.numBSAntennas,parIn.numOFDMSymbols);
for idxBlockAnt = 1:parIn.numBSAntennas
    YModDD((idxBlockAnt-1)*parIn.lenSprCodes+1:idxBlockAnt*parIn.lenSprCodes,:) = ...
    sample.YModNoise(:,:,idxBlockAnt);
end

noisePowerEst = noiseStdVarEst * noiseStdVarEst;
XModtildeEst = pinv(phiDDtilde'*phiDDtilde + noisePowerEst*eye(numActUEsEst))*phiDDtilde'*YModDD;
XModEst(idActUEsEst,:) = XModtildeEst;

result.chSpaEquEst(:,:,:,idxIter) = chSpaEquEst;
result.XModEst(:,:,idxIter) = XModEst;
end
result.actFlagEst = actFlagEst;
result.idActUEsEst = idActUEsEst;
end

