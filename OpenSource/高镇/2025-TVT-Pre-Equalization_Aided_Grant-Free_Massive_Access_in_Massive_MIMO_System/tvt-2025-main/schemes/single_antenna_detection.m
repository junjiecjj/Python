function [result] = single_antenna_detection(parIn,sample)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Single antenna detection scheme from Yikun Mei 
% 
% Coded by: Yueqing Wang, Beijing Institute of Technology
% Email: 17710820190@163.com
% Last change: 02/11/23
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% (norm(sample.YModNoiseEta - sample.sprCodes*sample.XMod,'fro').^2)/numel(sample.YModNoiseEta)
actFlagEst = zeros(parIn.numUEs,1);

%OMP
% [XModEst,index] = SOMP(sample.sprCodes, sample.YModNoiseEta, parIn.numActUEs);
[XModEst, index] = SOMP_noise(sample.sprCodes, sample.YModNoiseEta, parIn.varNoise);
actFlagEst(index) = 1;

% GSP
% [XModEst, supp, iter] = DMMV_SP(sample.YModNoiseEta, sample.sprCodes, parIn.numActUEs);
% gsp_ad = 0.01*max(max(max(abs(XModEst))));
% gsp_th = 0.9;
% actFlagEst(sum(sum(abs(XModEst)>gsp_ad,3),2)./parIn.numOFDMSymbols>= gsp_th) = 1; 

% Oracle LS
% sprcodes_oracle = sample.sprCodes(:,sample.idActUEs);
% XModEst_oracle = (sprcodes_oracle' * sprcodes_oracle)\(sprcodes_oracle'*sample.YModNoiseEta);
% XModEst = zeros(parIn.numUEs, parIn.numOFDMSymbols);
% XModEst(sample.idActUEs,:) = XModEst_oracle; 
% actFlagEst = sample.actFlag;

% OAMP
% [XModEstTemp, lambda] = OAMP_MMV_SSL(sample.YModNoiseEta, sample.sprCodes, parIn.dampOAMP, ...
%                                    parIn.numOAMPIters, parIn.priorOAMP, parIn.modOrder);
% 
% actFlagEst(mean(lambda,2) > parIn.adThOAMP) = 1;
% XModEst = XModEstTemp(:,:,parIn.numOAMPIters);

% [XModEst, lambda] = MMV_AMP_MPSK(sample.YModNoiseEta, sample.sprCodes, ...
%     parIn.dampAMP, 50, 1e-4, 0,parIn.modOrder);
% actFlagEst(mean(lambda,2) > parIn.adThOAMP) = 1;

result.XModEst = XModEst;
result.actFlagEst = actFlagEst;
result.idActUEsEst = find(actFlagEst==1);
end

