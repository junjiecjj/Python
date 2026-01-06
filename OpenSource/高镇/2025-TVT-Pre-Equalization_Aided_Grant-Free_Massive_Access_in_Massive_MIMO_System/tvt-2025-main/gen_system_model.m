function sample = gen_system_model(parIn)    

actFlag = zeros(parIn.numUEs,1);
idActUEs = randperm(parIn.numUEs,parIn.numActUEs);
idActUEs = sort(idActUEs);
actFlag(idActUEs) = 1;

% Generate Spreading Codes
if(parIn.scheme == 0)
    %sprTemp = dftmtx(parIn.numUEs)/sqrt(parIn.numUEs);
    sprCodes = sprTemp(randperm(parIn.numUEs,parIn.lenSprCodes),:);   % Partial DFT matrix
    % sprCodes = (randn(parIn.lenSprCodes,parIn.numUEs) + 1j*randn(parIn.lenSprCodes,parIn.numUEs))/sqrt(2); 
else
    %sprCodes = (randn(parIn.lenSprCodes,parIn.numUEs) + 1j*randn(parIn.lenSprCodes,parIn.numUEs))/sqrt(2*parIn.lenSprCodes); 
    sprCodes = (randn(parIn.lenSprCodes,parIn.numUEs) + 1j*randn(parIn.lenSprCodes,parIn.numUEs))/sqrt(2); 
end
% sprCodes = (randn(parIn.lenSprCodes,parIn.numUEs) + 1j*randn(parIn.lenSprCodes,parIn.numUEs))/sqrt(2); 

msgNum = randi([0 parIn.modOrder-1],parIn.numUEs,parIn.numOFDMSymbols);
msgMod = pskmod(msgNum, parIn.modOrder, 0, 'gray');
XMod = repmat(actFlag,1,parIn.numOFDMSymbols) .* msgMod;

% Genertate Channel
chSpa = zeros(parIn.numBSAntennas,parIn.lenSprCodes,parIn.numUEs);
chAng = zeros(parIn.numBSAntennas,parIn.lenSprCodes,parIn.numUEs);
for indexUser = 1:parIn.numUEs
    Lp = randi([parIn.Lp_min,parIn.Lp_max],1,1);
    [chSpaTemp, chAngTemp] = one_ring_channel(parIn.numBSAntennas, parIn.lenSprCodes,...
        parIn.lenSprCodes, parIn.bandWidth, parIn.fs, parIn.lenCP, parIn.pathGain,...
        Lp, parIn.channelTh,parIn.angleSpread);
    chSpa(:,:,indexUser) = chSpaTemp;
    chAng(:,:,indexUser) = chAngTemp;
end

% Pre-equalization
idPreEq = randi([1,parIn.numBSAntennas],1,1);
chTemp = squeeze(chSpa(idPreEq,:,:));
preEqMtx = (abs(chTemp)>parIn.preEqTh) .* (1./chTemp);
% preEqMtx = 1./chTemp;

% Equalized channel
preEqMtxTemp = reshape(preEqMtx,[1,size(preEqMtx,1),size(preEqMtx,2)]);
chSpaEqu = repmat(preEqMtxTemp,[parIn.numBSAntennas,1,1]) .* chSpa;

YBarMod = zeros(parIn.lenSprCodes,parIn.numOFDMSymbols,parIn.numBSAntennas);

for idxA = 1:parIn.numBSAntennas
    YBarMod(:,:,idxA) = sqrt(parIn.zeta)*(squeeze(chSpa(idxA,:,:)).*sprCodes.*preEqMtx) * XMod;
end
% [YModNoise,~,~] = awgn_new(YMod,parIn.snrdB,'measured');
% [YModNoiseOAMP,~,~] = awgn_new(YMod(:,:,idPreEq),parIn.snrdB,'measured');
awgn = sqrt(parIn.varNoise/2)*(randn(parIn.lenSprCodes,parIn.numOFDMSymbols,parIn.numBSAntennas) + ...
        1j*randn(parIn.lenSprCodes,parIn.numOFDMSymbols,parIn.numBSAntennas));
YBarModNoise = YBarMod + awgn;

YModNoise = sqrt(1/parIn.zeta)*YBarModNoise;

sample.sprCodes = sprCodes;
sample.chSpa = chSpa;
sample.chAng = chAng;
sample.chSpaEqu = chSpaEqu;
sample.msgNum = msgNum;
sample.XMod = XMod;
sample.YModNoise = YModNoise;
sample.YModNoiseEta = YModNoise(:,:,idPreEq);
%sample.YModNoiseOAMP = YModNoiseOAMP;
sample.idActUEs = idActUEs;
sample.actFlag = actFlag;

if(parIn.debugChannel)
    channelNorm = chTemp(:).*conj(chTemp(:));
    [f,xi] =ksdensity(channelNorm);
    plot(xi,f)
    preEqNorm = preEqMtx(:).*conj(preEqMtx(:));
    [f,xi] =ksdensity(preEqNorm);
    plot(xi,f)
end

if parIn.debugSNR 
    snr_preeq = var(YBarMod(:,:,idPreEq))/var(awgn(:,:,idPreEq));
    snr_average = var(YBarMod,1,'all')/var(awgn,1,'all');
    % PsEst_dBm = 10*log10(mean(YBarMod(:).*conj(YBarMod(:)))/(1e-3*parIn.numActUEs)) + parIn.pathLossEp_dBm;
    %powerAntenna = squeeze(mean(YModNoise.*conj(YModNoise),[1,2]));
    %figure;
    %stem(1:parIn.numBSAntennas, powerAntenna,'filled');
    %xlabel('Antenna index')
    %ylabel('Power of received signals')
end
sample.snr_preeq = snr_preeq;
sample.snr_average = snr_average;